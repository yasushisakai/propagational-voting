// Metal-backed squaring kernel for ppv.
//
// Matches the c_squaring ABI so ppv.py's ctypes wrapper can be reused with
// only the dylib path swapped:
//
//   int ppv_metal_squaring(
//       const float *Q, const float *R,
//       int ndi, int num_delegates, int num_policies,
//       float tol, int max_iter,
//       float *consensus_out, float *influence_out);
//
// Algorithm is the same joint squaring recurrence:
//
//   P_m = Q^(2^m)            T_m = I + Q + ... + Q^(2^m - 1)
//   T_{m+1} = T_m + T_m·P_m   P_{m+1} = P_m·P_m
//
// The two GEMMs each iteration go to the GPU via MPSMatrixMultiplication.
// CPU does the cheap stuff: max reduction (convergence check), the daxpy
// (T += scratch), and the final output matvecs. Shared-storage MTLBuffers
// give us zero-copy access between CPU and GPU on Apple Silicon.

import Metal
import MetalPerformanceShaders
import Accelerate

@_cdecl("ppv_metal_squaring")
public func ppv_metal_squaring(
    Q: UnsafePointer<Float>,
    R: UnsafePointer<Float>,
    ndi: Int32,
    numDelegates: Int32,
    numPolicies: Int32,
    tol: Float,
    maxIter: Int32,
    consensusOut: UnsafeMutablePointer<Float>,
    influenceOut: UnsafeMutablePointer<Float>
) -> Int32 {
    // Called from Python via ctypes: there is no AppKit runloop draining the
    // global autorelease pool between calls, so Metal objects (MTLBuffer, MPS
    // wrappers, MTLCommandBuffer) would accumulate across cells until the
    // GPU runs out of memory or hangs. Drain explicitly here.
    return autoreleasepool { () -> Int32 in
        return ppvMetalSquaringImpl(
            Q: Q, R: R, ndi: ndi,
            numDelegates: numDelegates, numPolicies: numPolicies,
            tol: tol, maxIter: maxIter,
            consensusOut: consensusOut, influenceOut: influenceOut
        )
    }
}

private func ppvMetalSquaringImpl(
    Q: UnsafePointer<Float>,
    R: UnsafePointer<Float>,
    ndi: Int32,
    numDelegates: Int32,
    numPolicies: Int32,
    tol: Float,
    maxIter: Int32,
    consensusOut: UnsafeMutablePointer<Float>,
    influenceOut: UnsafeMutablePointer<Float>
) -> Int32 {
    let n = Int(ndi)
    let n2 = n * n
    let bytes = n2 * MemoryLayout<Float>.stride

    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        return -1
    }

    // Allocate four shared-storage buffers: t (accumulator), p (current power),
    // scratch (t@p output), pNew (p@p output, so we don't race scratch).
    let opts: MTLResourceOptions = .storageModeShared
    guard let tBuf = device.makeBuffer(length: bytes, options: opts),
          let pBuf = device.makeBuffer(length: bytes, options: opts),
          let scratchBuf = device.makeBuffer(length: bytes, options: opts),
          let pNewBuf = device.makeBuffer(length: bytes, options: opts) else {
        return -1
    }

    // Initial values on the shared buffers (CPU writes, GPU will read).
    let tPtr = tBuf.contents().assumingMemoryBound(to: Float.self)
    let pPtr = pBuf.contents().assumingMemoryBound(to: Float.self)
    let scratchPtr = scratchBuf.contents().assumingMemoryBound(to: Float.self)
    memset(tBuf.contents(), 0, bytes)
    for i in 0..<n { tPtr[i * n + i] = 1.0 }
    memcpy(pBuf.contents(), Q, bytes)

    let desc = MPSMatrixDescriptor(
        rows: n, columns: n,
        rowBytes: n * MemoryLayout<Float>.stride,
        dataType: .float32
    )

    // Pointers to the MTLBuffer wrappers we swap in the inner loop.
    var pCurrent = pBuf
    var pNext = pNewBuf

    let mmul = MPSMatrixMultiplication(
        device: device,
        transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: n, interiorColumns: n,
        alpha: 1.0, beta: 0.0
    )

    var iters: Int32 = 0
    var converged = false
    for m in 0..<Int(maxIter) {
        if converged { break }
        // Per-iteration autoreleasepool drains MPSMatrix wrappers and the
        // command buffer's intermediate ObjC objects. Without this, large-n
        // runs accumulate enough Metal state to hang the GPU.
        autoreleasepool {
            // Convergence check on the CURRENT p (CPU-side, shared memory).
            var pMax: Float = 0
            let curPtr = pCurrent.contents().assumingMemoryBound(to: Float.self)
            vDSP_maxv(curPtr, 1, &pMax, vDSP_Length(n2))
            if pMax < tol { iters = Int32(m); converged = true; return }

            let tMat = MPSMatrix(buffer: tBuf, descriptor: desc)
            let pMat = MPSMatrix(buffer: pCurrent, descriptor: desc)
            let scratchMat = MPSMatrix(buffer: scratchBuf, descriptor: desc)
            let pNextMat = MPSMatrix(buffer: pNext, descriptor: desc)

            guard let cmd = queue.makeCommandBuffer() else {
                iters = -1; converged = true; return
            }
            mmul.encode(commandBuffer: cmd,
                        leftMatrix: tMat, rightMatrix: pMat,
                        resultMatrix: scratchMat)
            mmul.encode(commandBuffer: cmd,
                        leftMatrix: pMat, rightMatrix: pMat,
                        resultMatrix: pNextMat)
            cmd.commit()
            cmd.waitUntilCompleted()

            vDSP_vadd(tPtr, 1, scratchPtr, 1, tPtr, 1, vDSP_Length(n2))
            let tmp = pCurrent; pCurrent = pNext; pNext = tmp
            iters = Int32(m + 1)
        }
    }

    // Outputs from t (= T_∞ ≈ (I - Q)^-1 on the transient block).
    // Done on CPU: cheap, and t is already in shared memory.
    let nd = Int(numDelegates)
    let np = Int(numPolicies)

    // consensus = R · t · e_d, where e_d = [1]*nd + [0]*(n-nd).
    // Compute as two matvecs.
    var eD = [Float](repeating: 0, count: n)
    for i in 0..<nd { eD[i] = 1.0 }
    var tEd = [Float](repeating: 0, count: n)
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        Int32(n), Int32(n),
        1.0, tPtr, Int32(n), eD, 1,
        0.0, &tEd, 1
    )
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        Int32(np), Int32(n),
        1.0, R, Int32(n), tEd, 1,
        0.0, consensusOut, 1
    )

    // influence_i = row_sum(t)_i / diag(t)_i
    var ones = [Float](repeating: 1.0, count: n)
    var rowSums = [Float](repeating: 0, count: n)
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        Int32(n), Int32(n),
        1.0, tPtr, Int32(n), ones, 1,
        0.0, &rowSums, 1
    )
    for i in 0..<n {
        influenceOut[i] = rowSums[i] / tPtr[i * n + i]
    }

    return iters
}
