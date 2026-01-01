---
layout: post
title:  "Making CPUs adequate at linear algebra"
date:   2025-12-30 00:00:01
categories:
---

$$\require{amsmath}$$

If you are computing vector/matrix operations like addition or multiplication, you're probably using a GPU to get the job done.
I'm also counting the hundred or so other inference card types here as well, meaning your TPUs, ANEs, FPGAs, ASICs, LPUs, and so on.
This is usually the right choice, but there are exceptions that make a compelling case for using a CPU instead.
For an example of such an scenario, you can see [here](./when-you-should-not-use-a-gpu-to-run-a-machine-learning-model).
The purpose of this post is to explore some of the techical details related to extracting the most performance out of your CPU in such situations.

This post assumes prior knowledge of the x86 instruction set extension [AVX-512](https://en.wikipedia.org/wiki/AVX-512) and its intrinsic functions.
If you don't have prior familiarity with this subject, you can read my [other post](./the-anatomy-of-avx-512-intrinsics) to fill in the gaps.
The main point of this post will be writing code using intrinsics, and we will deal with assembly only when it has been produced by the compiler.
We're only really looking at relative performance instead of absolute performance, so the specifics of the setup aren't critical to the message as long as the CPU supports AVX-512.
However, for the sake of completeness, the tests will be run on a Ryzen 9 7900X, and the code will be written in C++ before being compiled with gcc 15.2.1.

I will omit all thread-level parallelization, as the point of this post is to capture the computational potential of a single core.
This doesn't mean you shouldn't add thread-level parallelization on top of everything that's discussed in this post.
When doing something like this in a real application, you definitely **should** explore if that would be a good idea.

# The obvious way
Matrix and vector operations are all just some combination of run-of-the-mill scalar operations.
To compute the sum of two vectors, you just sum their individual elements.
For a dot product, you instead multiply their elements and sum the results.
A matrix-matrix sum is the same as a vector sum, whereas matrix-vector multiplication entails just taking the dot product between the vector and all of the rows of the matrix.
Matrix-matrix multiplication just requires taking dot products between the rows/columns of the matrices being multiplied.
As noted, these dot products are nothing more than some scalar produts and sums.
It's just sums and products all the way down.
Nothing more, nothing less.

As such, the obvious way to do any one of these matrix or vector operations is to just, you know, calculate the individual sums and products.
One by one.
These operations are going to happen anyways no matter how you phrase things, so it's not like we're doing unnecessary work.
This will work, of course, but the performance penalty for doing this in an ordinary loop will range from massive to an order of magnitude.
Let's dive in.

# The starting point
The simplest example I could come up with is a dot product between two large vectors.
We'll use this as a starting point.
Below is the boilerplate that will be used for these first experiments.
Take note of what ceil_to_multiple and `u32` are defined as.
They will be reused in all future examples, but will be omitted from the code shown.

{% highlight c++ %}
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <random>

using u32 = std::uint32_t;

u32 ceil_to_multiple(u32 n, u32 base) { return (n + base - 1) / base * base; }

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " method\n";
    return 1;
  }

  std::string method = argv[1];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  const u32 n = /* number of elements */;

  float *a = new float[n];
  float *b = new float[n];

  const u32 padded_size = ceil_to_multiple(n, 16);
  float *pa = new float[padded_size];
  float *pb = new float[padded_size];

  for (u32 i = 0; i < n; ++i) {
    a[i] = pa[i] = dis(gen);
    b[i] = pb[i] = dis(gen);
  }

  for (u32 i = n; i < padded_size; ++i) {
    pa[i] = 0.0f;
    pb[i] = 0.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();

  float result = -1;
  if (method == "naive") {
    result = /* naive function */(a, b, padded_size);
  } else if (method == "avx512") {
    result = /* AVX-512 function */(pa, pb, padded_size);
  } else {
    delete[] a;
    delete[] b;
    delete[] pa;
    delete[] pb;

    std::cerr << "Invalid method, expected naive or avx512, got " << method
              << "\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // Without this, the compiler just skips the calculations :)
  std::cout << result << std::endl;

  delete[] a;
  delete[] b;
  delete[] pa;
  delete[] pb;

  return 0;
}
{% endhighlight %}

The arrays are initialized with random values sampled from a uniform distribution in the range (-1, 1).
We form both the unpadded and the padded arrays (for the naive and AVX-512 implementation) before calling the function, with both arrays containing the same non-zero values.
This is wasteful, but I found this code to be the easiest to read and it allows us to simplify the code in the dot product functions themselves.
This makes it a bit less of a chore to decipher their compiled versions.
As the padding is done with zeroes, it will not affect the resulting value.
You might be wondering how much we're scewing the results by not counting the time it takes to pad the arrays or to compute the final multiplications that couldn't fill an entire 512-bit register naively.
I went ahead and checked; this doesn't change the results in any meaningful way.
There is also a nicer way to do the padding that will be shown later, but we're aiming for simplicity here in the first example.

As mentioned earlier, the dot product requires us to do an element-wise multiplication and then summing all of the resulting values together.
The vectors must be the same size for a dot product to be well defined.
For two $n$-sized vectors, we then have to perform $n$ multiplications and $n-1$ additions for a total of $2n-1$ scalar operations.
With $n=3$, we have to do $5$ operations.
With $n=3000$, it's $5999$ operations.
The increase in number of operations is linear with respect to the sizes of the vectors, so increasing the vector size will not cause the number of operations to explode.

Let's get the naive implementation out of the way first.
Here's the code:
{% highlight c++ %}
float dot_product_naive(const float *a, const float *b, u32 n) {
  float sum = 0.0f;
  for (u32 i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
{% endhighlight %}

We take in two arrays of floats and their respective size, iterate over each element index and multiply the respective values.
This product is then summed into an accumulator variable, which is returned after the loop has finished.

# A quick peek at the assembly
With optimizations off, this is what the function complies into:

{% highlight nasm %}
dot_product_naive(float const*, float const*, int):
        push    rbp
        mov     rbp, rsp
        mov     QWORD PTR [rbp-24], rdi
        mov     QWORD PTR [rbp-32], rsi
        mov     DWORD PTR [rbp-36], edx
        pxor    xmm0, xmm0
        movss   DWORD PTR [rbp-4], xmm0
        mov     DWORD PTR [rbp-8], 0
        jmp     .L2
.L3:
        mov     eax, DWORD PTR [rbp-8]
        cdqe
        lea     rdx, [0+rax*4]
        mov     rax, QWORD PTR [rbp-24]
        add     rax, rdx
        movss   xmm1, DWORD PTR [rax]
        mov     eax, DWORD PTR [rbp-8]
        cdqe
        lea     rdx, [0+rax*4]
        mov     rax, QWORD PTR [rbp-32]
        add     rax, rdx
        movss   xmm0, DWORD PTR [rax]
        mulss   xmm0, xmm1
        movss   xmm1, DWORD PTR [rbp-4]
        addss   xmm0, xmm1
        movss   DWORD PTR [rbp-4], xmm0
        add     DWORD PTR [rbp-8], 1
.L2:
        mov     eax, DWORD PTR [rbp-8]
        cmp     eax, DWORD PTR [rbp-36]
        jl      .L3
        movss   xmm0, DWORD PTR [rbp-4]
        pop     rbp
        ret
{% endhighlight %}

The first block just sets the correct initial register values for the variables  
* **sum** (rbp-4)  
* **i** (rbp-8)   
* **n** (rbp-36)

and the pointers  
* **a** (rbp-24) and  
* **b** (rbp-32).

Jumping to .L2, there is a check (cmp) on whether $i < n$.
If it's true, then execution jumps to .L3 (jl stands for "jump if less than").
The .L3 block does some housekeeping in order to have the values ready for multiplication, after which we have the key lines:
{% highlight nasm %}
; At this point, xmm0 = a[i] and xmm1 = b[i]
mulss   xmm0, xmm1                      ; xmm0 = xmm0 * xmm1
movss   xmm1, DWORD PTR [rbp-4]         ; xmm1 = sum
addss   xmm0, xmm1                      ; xmm0 = xmm0 + xmm1
movss   DWORD PTR [rbp-4], xmm0         ; sum = xmm0
{% endhighlight %}

Then, $i$ is incremented with the add instruction and the execution falls through.
Execution picks up again from the beginning of the .L2 block.
This cycle then continues until $i = n$, after which our sum variable is written into the `xmm0` register and the function returns.

It's worthwhile to realize that we are using `xmm` registers, which refer to 128-bit SIMD registers introduced by the Streaming SIMD Extensions (SSE) instruction set.
These registers are often used by compilers because SSE2 support (and therefore the existence of these registers) is guaranteed on any and all x86-64 CPUs.
This way, the compiled program is going to run without problems on any 64-bit PC.
But don't get confused by the use of these registers, as the algebraic operations themselves are still working with scalar data.
This is evident by the usage of `mulss` instead of `mulps`, and `addss` instead of `addps`.
There are operating on scalar (`s`) and not packed (`p`) data, as was the naming convention in SSE.
So the compiled program is in fact using registers first introduced by a SIMD instruction set extension that **could** be used for vectorization, but it is not actually using them for that.

# What's the runtime?

Let's get a baseline on performance with this.
When running the function with two vectors containing a billion elements each, $n = 1 000 000 000$, we get a runtime of around 2.21 seconds.
This is not bad, given that we are calculating nearly two billion scalar operations using floating point values.
As we've seen, our compiled binary also consists of continuously running the same 20 instructions in order to jump between the .L2 and .L3 blocks, only achieving a single scalar multiplication and sum with each passing iteration.

# Turning up the optimizations

Thankfully, modern compilers are more than capable of optimizing code like this. 
This means that we should be able to get more performance for free by just enabling optimizations.
We're going to do that from now on by enabling the gcc flags `-mavx512f -mavx512vl -march=native -O3 -ftree-vectorize -funroll-loops`.
The first three are related to the AVX-512 instructions we will be using later, while the latter three activate optimizations that might be useful for the naive implementation as well.

The assembly becomes quite a bit more complicated after doing this, so we won't go over it in as much detail.
However, now that we have the orientation behind us, you can give reading it a shot if you wish.
Here's what the function compiles into:

{% highlight nasm %}
dot_product_naive(float const*, float const*, int):
        mov     ecx, edx
        test    edx, edx
        jle     .L9
        lea     eax, [rdx-1]
        cmp     eax, 14
        jbe     .L10
        mov     r8d, edx
        xor     r9d, r9d
        vxorps  xmm0, xmm0, xmm0
        shr     r8d, 4
        sal     r8, 6
.L4:
        vmovups zmm1, ZMMWORD PTR [rsi+r9]
        vmulps  zmm5, zmm1, ZMMWORD PTR [rdi+r9]
        add     r9, 64
        vaddss  xmm0, xmm0, xmm5
        vshufps xmm6, xmm5, xmm5, 85
        vunpckhps       xmm8, xmm5, xmm5
        vshufps xmm10, xmm5, xmm5, 255
        vextractf32x4   xmm12, ymm5, 1
        valignd ymm14, ymm5, ymm5, 5
        valignd ymm1, ymm5, ymm5, 6
        vaddss  xmm7, xmm0, xmm6
        valignd ymm2, ymm5, ymm5, 7
        vextracti64x4   ymm5, zmm5, 0x1
        vaddss  xmm9, xmm7, xmm8
        vshufps xmm8, xmm5, xmm5, 85
        vaddss  xmm11, xmm9, xmm10
        vunpckhps       xmm10, xmm5, xmm5
        vaddss  xmm13, xmm11, xmm12
        vshufps xmm12, xmm5, xmm5, 255
        vaddss  xmm15, xmm13, xmm14
        vextractf32x4   xmm14, ymm5, 1
        vaddss  xmm3, xmm15, xmm1
        valignd ymm1, ymm5, ymm5, 5
        vaddss  xmm4, xmm3, xmm2
        valignd ymm2, ymm5, ymm5, 6
        vaddss  xmm7, xmm4, xmm5
        valignd ymm5, ymm5, ymm5, 7
        vaddss  xmm9, xmm7, xmm8
        vaddss  xmm11, xmm9, xmm10
        vaddss  xmm13, xmm11, xmm12
        vaddss  xmm15, xmm13, xmm14
        vaddss  xmm3, xmm15, xmm1
        vaddss  xmm4, xmm3, xmm2
        vaddss  xmm0, xmm4, xmm5
        cmp     r8, r9
        jne     .L4
        mov     r10d, ecx
        and     r10d, -16
        mov     edx, r10d
        cmp     ecx, r10d
        je      .L21
.L3:
        mov     r11d, ecx
        sub     r11d, edx
        lea     eax, [r11-1]
        cmp     eax, 6
        jbe     .L7
        vmovups ymm6, YMMWORD PTR [rsi+rdx*4]
        vmulps  ymm7, ymm6, YMMWORD PTR [rdi+rdx*4]
        vaddss  xmm0, xmm0, xmm7
        vshufps xmm10, xmm7, xmm7, 85
        vunpckhps       xmm12, xmm7, xmm7
        vshufps xmm14, xmm7, xmm7, 255
        vextractf32x4   xmm1, ymm7, 1
        valignd ymm2, ymm7, ymm7, 5
        valignd ymm5, ymm7, ymm7, 6
        vaddss  xmm11, xmm0, xmm10
        valignd ymm7, ymm7, ymm7, 7
        vaddss  xmm13, xmm11, xmm12
        vaddss  xmm15, xmm13, xmm14
        vaddss  xmm3, xmm15, xmm1
        vaddss  xmm4, xmm3, xmm2
        vaddss  xmm6, xmm4, xmm5
        vaddss  xmm0, xmm6, xmm7
        test    r11b, 7
        je      .L21
        and     r11d, -8
        add     r10d, r11d
.L7:
        movsx   rdx, r10d
        lea     r8d, [r10+1]
        vmovss  xmm8, DWORD PTR [rdi+rdx*4]
        vfmadd231ss     xmm0, xmm8, DWORD PTR [rsi+rdx*4]
        cmp     ecx, r8d
        jle     .L21
        lea     r9d, [r10+2]
        vmovss  xmm9, DWORD PTR [rsi+4+rdx*4]
        vfmadd231ss     xmm0, xmm9, DWORD PTR [rdi+4+rdx*4]
        cmp     ecx, r9d
        jle     .L21
        lea     r11d, [r10+3]
        vmovss  xmm10, DWORD PTR [rdi+8+rdx*4]
        vfmadd231ss     xmm0, xmm10, DWORD PTR [rsi+8+rdx*4]
        cmp     ecx, r11d
        jle     .L21
        lea     eax, [r10+4]
        vmovss  xmm11, DWORD PTR [rdi+12+rdx*4]
        vfmadd231ss     xmm0, xmm11, DWORD PTR [rsi+12+rdx*4]
        cmp     ecx, eax
        jle     .L21
        lea     r8d, [r10+5]
        vmovss  xmm12, DWORD PTR [rdi+16+rdx*4]
        vfmadd231ss     xmm0, xmm12, DWORD PTR [rsi+16+rdx*4]
        cmp     ecx, r8d
        jle     .L21
        add     r10d, 6
        vmovss  xmm13, DWORD PTR [rdi+20+rdx*4]
        vfmadd231ss     xmm0, xmm13, DWORD PTR [rsi+20+rdx*4]
        cmp     ecx, r10d
        jle     .L21
        vmovss  xmm14, DWORD PTR [rdi+24+rdx*4]
        vfmadd231ss     xmm0, xmm14, DWORD PTR [rsi+24+rdx*4]
        vzeroupper
        ret
.L21:
        vzeroupper
        ret
.L9:
        vxorps  xmm0, xmm0, xmm0
        ret
.L10:
        xor     edx, edx
        xor     r10d, r10d
        vxorps  xmm0, xmm0, xmm0
        jmp     .L3
{% endhighlight %}

An important thing to note here is the register names.
Previously, we were exclusively using XMM registers, which are 128 bits wide.
Here, we see YMM registers (256 bits) and even ZMM registers (512 bits) make their appearance.
So we're definitely doing something that utilizes our CPU's modern SIMD registers.

A second thing to note is that we have a somewhat recognizable block structure, but the instructions inside the blocks often start with the letter *v*.
So for instance, an instruction like `addss` is no longer present, but the instruction `vaddss` has appeared in its place.
These are instructions from the AVX instruction set, as opposed to the SSE, and **are** in fact utilizing vectorization.
For instance, there's two instances of `vmulps`: one with ZMM (512-bit) registers and a second one with YMM (256-bit) registers.
This instruction performs vectorized element-wise multiplication on its operands.
As we're using 32-bit floats, there's 16 of them in each ZMM register and 8 in each YMM register.
This means that we're calculating 16 scalar multiplications in the first, and 8 scalar multiplicatins in the second case, both with just one instruction.
This looks promising.

In fact, our runtime drops to $0.66$ seconds without changing a single line of code.
Clearly, the compiler has done a very nice job.

# The order of operations
As is evident, if you had to optimize code like this the first step would definitely be to just add all the relevant compiler optimization flags.
If the runtime is acceptable after doing so, great, you didn't have to allocate any developer time to optimizing the code itself.
If the resulting runtime is still not acceptable, things get interesting.
The order of operations, in my opinion, should go like this:
1. See if you can find a really quick and easy way to re-arrange your code in a way that makes the compiler's job easier, then try compiling it again. 
2. Check if a library like [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) has support for whatever it is you're trying to do.
3. Rewrite your code using intrinsics.
4. Begin to fiddle with the resulting assembly yourself.

Let's assume we've now enabled optimizations, can't come up with any clever ways to reorganize our code and BLAS libraries don't have what we need.
We're still not happy with the runtime, so we start looking at intrinsics.

# Intrinsics
Here is one way to rewrite the previous implementation using AVX-512 intrinsics:

{% highlight c++ %}
float dot_product_avx512(float *a, float *b, u32 n) {
  __m512 sum = _mm512_setzero_ps();

  for (u32 i = 0; i < n; i += 16) {
    __m512 va = _mm512_loadu_ps(&a[i]);
    __m512 vb = _mm512_loadu_ps(&b[i]);
    sum = _mm512_fmadd_ps(va, vb, sum);
  }

  return _mm512_reduce_add_ps(sum);
}
{% endhighlight %}

We have gone from 5 lines of code to 7 lines of code, but things have certainly gotten a lot more complicated, at least conceptually speaking.
As mentioned at the beginning of the post, if you're not yet able to read the above program, you can read my [other post](./the-anatomy-of-avx-512-intrinsics) that outlines the process.

In this version, we first initialize an accumulator register with zeroes.
Then, we're assuming the arrays are of an equal size that is divisible by 16 (32-bit floats, so 512 bits) and start iterating in such increments.
For each batch, 512 bits of data from both arrays is loaded into their respective registers, whose element-wise product is summed element-wise into the accumulator register using the FMA instruction.
This way, we are left with 16 floats in our accumulator register at the end, which we can sum together as a final step before returning.
Here's the compiled assembly:
{% highlight nasm %}
dot_product_avx512(float*, float*, unsigned int):
        mov     ecx, edx
        test    edx, edx
        je      .L4
        vmovups zmm2, ZMMWORD PTR [rdi]
        lea     edx, [rdx-1]
        vxorps  xmm0, xmm0, xmm0
        mov     eax, 16
        shr     edx, 4
        vfmadd231ps     zmm0, zmm2, ZMMWORD PTR [rsi]
        and     edx, 7
        cmp     eax, ecx
        jnb     .L2
        test    edx, edx
        je      .L3
        cmp     edx, 1
        je      .L29
        cmp     edx, 2
        je      .L30
        cmp     edx, 3
        je      .L31
        cmp     edx, 4
        je      .L32
        cmp     edx, 5
        je      .L33
        cmp     edx, 6
        je      .L34
        vmovups zmm3, ZMMWORD PTR [rdi+64]
        mov     eax, 32
        vfmadd231ps     zmm0, zmm3, ZMMWORD PTR [rsi+64]
.L34:
        mov     r8d, eax
        add     eax, 16
        vmovups zmm6, ZMMWORD PTR [rdi+r8*4]
        vfmadd231ps     zmm0, zmm6, ZMMWORD PTR [rsi+r8*4]
.L33:
        mov     r9d, eax
        add     eax, 16
        vmovups zmm7, ZMMWORD PTR [rdi+r9*4]
        vfmadd231ps     zmm0, zmm7, ZMMWORD PTR [rsi+r9*4]
.L32:
        mov     r10d, eax
        add     eax, 16
        vmovups zmm1, ZMMWORD PTR [rdi+r10*4]
        vfmadd231ps     zmm0, zmm1, ZMMWORD PTR [rsi+r10*4]
.L31:
        mov     r11d, eax
        add     eax, 16
        vmovups zmm4, ZMMWORD PTR [rdi+r11*4]
        vfmadd231ps     zmm0, zmm4, ZMMWORD PTR [rsi+r11*4]
.L30:
        mov     edx, eax
        add     eax, 16
        vmovups zmm5, ZMMWORD PTR [rdi+rdx*4]
        vfmadd231ps     zmm0, zmm5, ZMMWORD PTR [rsi+rdx*4]
.L29:
        mov     r8d, eax
        add     eax, 16
        vmovups zmm8, ZMMWORD PTR [rdi+r8*4]
        vfmadd231ps     zmm0, zmm8, ZMMWORD PTR [rsi+r8*4]
        cmp     eax, ecx
        jnb     .L2
.L3:
        mov     r9d, eax
        lea     r10d, [rax+16]
        lea     r11d, [rax+32]
        vmovups zmm9, ZMMWORD PTR [rdi+r9*4]
        vmovups zmm10, ZMMWORD PTR [rdi+r10*4]
        lea     edx, [rax+48]
        lea     r8d, [rax+64]
        vmovups zmm11, ZMMWORD PTR [rdi+r11*4]
        vmovups zmm12, ZMMWORD PTR [rdi+rdx*4]
        vfmadd231ps     zmm0, zmm9, ZMMWORD PTR [rsi+r9*4]
        vmovups zmm13, ZMMWORD PTR [rdi+r8*4]
        lea     r9d, [rax+80]
        vmovups zmm14, ZMMWORD PTR [rdi+r9*4]
        vfmadd231ps     zmm0, zmm10, ZMMWORD PTR [rsi+r10*4]
        lea     r10d, [rax+96]
        vmovups zmm15, ZMMWORD PTR [rdi+r10*4]
        vfmadd231ps     zmm0, zmm11, ZMMWORD PTR [rsi+r11*4]
        lea     r11d, [rax+112]
        sub     eax, -128
        vmovups zmm2, ZMMWORD PTR [rdi+r11*4]
        vfmadd231ps     zmm0, zmm12, ZMMWORD PTR [rsi+rdx*4]
        vfmadd231ps     zmm0, zmm13, ZMMWORD PTR [rsi+r8*4]
        vfmadd231ps     zmm0, zmm14, ZMMWORD PTR [rsi+r9*4]
        vfmadd231ps     zmm0, zmm15, ZMMWORD PTR [rsi+r10*4]
        vfmadd231ps     zmm0, zmm2, ZMMWORD PTR [rsi+r11*4]
        cmp     eax, ecx
        jb      .L3
.L2:
        vextractf64x4   ymm3, zmm0, 0x1
        vaddps  ymm0, ymm3, ymm0
        vextractf32x4   xmm6, ymm0, 0x1
        vaddps  xmm7, xmm6, xmm0
        vpermilps       xmm1, xmm7, 78
        vaddps  xmm4, xmm1, xmm7
        vshufps xmm8, xmm4, xmm4, 85
        vaddss  xmm0, xmm4, xmm8
        vzeroupper
        ret
.L4:
        vxorpd  xmm0, xmm0, xmm0
        jmp     .L2
{% endhighlight %}

The main computational burden comes from the `vfmadd231ps` instructions, which were notably missing in the previous version.
This instruction is carrying out the the FMA operation, it's taking `zmm0` (our accumulator) and summing into it the product of the two other registers.
As you may remember, this was performed using two different instructions in the previous version, namely `vmulps` and `vaddss`.
It seems that the program is also performing five of these instructions in a row and not just one per block.
Looks clever.
There's many other differences as well, the programs are actually vastly different when you start looking closely.

One would hope that these vast differences would then lead to some measurable speedup as well, and in fact they do, our runtime drops to 0.16 seconds.
This means we've achieved a 4.2x speedup by using intrinsics.

# Image processing
Of course, we can throw out toy examples all day long.
Although I have no intention of writing some fully fledged software suite for the purposes of this post, we definitely need something more substantial.
Here's an idea: brightening an image.
A standard image saved on disk as a jpg has three channels, one for each color channel.
A png will have four, but the last one is just the opacity and we don't need to change that.
As such, to brighten an image we can just take the first three channels of an image and add some constant to all its pixel intensities.
I'll save you the code related to loading and saving images in C++, but you can find it in the Github repository linked at the bottom of the post if you are interested.
Here's both a naive and an AVX-512 implementation for the brightening function:

{% highlight c++ %}
void brighten_naive(uint8_t *data, int w, int h, int c, int brightness) {
  size_t total = w * h * c;
  for (size_t i = 0; i < total; ++i) {
    int v = data[i] + brightness;
    data[i] = (uint8_t)std::clamp(v, 0, 255);
  }
}

void brighten_avx512(uint8_t *data, int w, int h, int c, int brightness) {
  size_t total = w * h * c;
  size_t i = 0;

  // Create the 512 bit vector that will be added to each "batch" of values
  // in the image.
  __m512i addV = _mm512_set1_epi8((signed char)brightness);

  for (; i + 64 <= total; i += 64) {
    __m512i batch = _mm512_loadu_si512((__m512i *)&data[i]);
    __m512i result = _mm512_adds_epu8(batch, addV);
    _mm512_storeu_si512((__m512i *)&data[i], result);
  }

  size_t remaining = total - i;
  if (remaining > 0) {
    // For example, with remaining == 2
    // 0000 0001 -> 0000 0100 -> 0000 0011
    __mmask16 mask = (1ULL << remaining) - 1;

    // Load the data into a register with the mask, other elements are zeroed
    // out.
    __m512i tail = _mm512_maskz_loadu_epi8(mask, &data[i]);

    // Calculate the saturated add on the data.
    __m512i res = _mm512_adds_epu8(tail, addV);

    // Write the result from the register back into the data array using the
    // same mask, so that we don't overwrite prior values with zeroes.
    _mm512_mask_storeu_epi8(&data[i], mask, res);
  }
}

{% endhighlight %}

Hopefully the implementations are clear from the code and their accompanying comments.
As we're working with a data type that is just 8 bits wide, we can increment by 64 elements in the loop.
I've also taken this chance to show an alternative way to handle data whose size is not divisible by 512 bits.
As is evident, masks provide a natural way to do this without having to pad the data.
We're also making use of saturated adds, meaning that our pixel intensities won't wrap around to 0 if we overflow.
Instead, we will just get a full-intensity color channel with a value of 255.
We're going to jump directly into comparing the naive function using compiler optimizations, and the function using intrinsics.
I'll be using a 960x1290 image and running the function repeatedly on it 10 000 times to get a measurable difference between the functions.
This would be imitating us doing this for 10 000 different images that we need to brighten, or a few really large images.
This of course turns the image completely white, but I'll assume that this doesn't change the computational complexity.
Here are the runtimes:

* Naive: 1.70 seconds
* AVX-512: 0.44 seconds

Again, around a 4x speedup from using intrinsics. 

# Neural networks
As a final example, let's look at the forward pass of a neural network's linear layer.
This one is going to be a full class implementation, and I'll make the entire program visible to make things clear.
Here's the code:

{% highlight c++ %}
template <u32 InDims, u32 OutDims> class LinearLayer {
public:
  std::random_device rd;

  // Number of input/output dimensions
  static constexpr u32 input_size = InDims;
  static constexpr u32 output_size = OutDims;
  static constexpr u32 padded_output_size = ceil_to_multiple(OutDims, 16);

  // Initialize weights and biases
  void write_parameters() {
    std::mt19937 gen(this->rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (u32 i = 0; i < padded_output_size; ++i) {
      for (u32 j = 0; j < this->input_size; ++j) {

        if (i >= output_size) {
          // This is a padding weight
          weights[j * output_size + i] = 0.0f;
        } else {
          weights[j * output_size + i] = dis(gen);
        }
      }

      if (i >= output_size) {
        // This is a padding bias
        bias[i] = 0.0f;
      } else {
        bias[i] = dis(gen);
      }
    }
  }

  void forward_naive(float *input, float *output) {
    for (u32 i = 0; i < output_size; ++i) {
      float sum = 0.0f;
      for (u32 j = 0; j < input_size; ++j) {
        sum += input[j] * weights[j * this->output_size + i];
      }
      output[i] = sum + bias[i];
    }
  }

  void forward_avx512(const float *input, float *output) {
    constexpr int widthOutput = padded_output_size / 16;

    __m512 res[widthOutput];
    for (int i = 0; i < widthOutput; i++) {
      res[i] = _mm512_loadu_ps(&bias[i * 16]);
    }

    for (int j = 0; j < this->input_size; ++j) {
      __m512 input_neuron = _mm512_set1_ps(input[j]);

      for (int i = 0; i < widthOutput; ++i) {
        __m512 curr_weights =
            _mm512_loadu_ps(&weights[j * this->output_size + i * 16]);

        res[i] = _mm512_fmadd_ps(input_neuron, curr_weights, res[i]);
      }
    }
    for (int i = 0; i < widthOutput; i++) {
      _mm512_storeu_ps(&output[i * 16], res[i]);
    }
  }
  ~LinearLayer() {
    delete[] weights;
    delete[] bias;
  }

private:
  float *weights = new float[input_size * padded_output_size];
  float *bias = new float[padded_output_size];
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " method\n";
    return 1;
  }
  std::string method = argv[1];

  const u32 input_size = 1024;
  const u32 output_size = 512;

  LinearLayer<input_size, output_size> layer;
  layer.write_parameters();

  float *a = new float[input_size];
  float *b = new float[output_size];

  float *pb = new float[layer.padded_output_size];

  for (int i = 0; i < input_size; ++i) {
    a[i] = 1.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10000; ++i) {
    if (method == "naive") {
      layer.forward_naive(a, b);
    } else if (method == "avx512") {
      layer.forward_avx512(a, pb);
    } else {
      delete[] a;
      delete[] b;
      delete[] pb;
      std::cerr << "Invalid method, expected naive or avx512, got " << method
                << "\n";
      return 1;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  delete[] a;
  delete[] b;
  delete[] pb;

  return 0;
}
{% endhighlight %}

We have a single linear layer that works with 32-bit floats.
It's input dimensionality is 1024 and output dimensionality 512.
We are passing in a vector of ones, but the weights and biases are initialized from a uniform distribution in the range (-1, 1).
Again, we are running the forward pass 10 000 times to get a measurable difference.
This would be like a forward pass of a ten-layer network a thousand times, or a forward pass of a one-layer network for 10 000 different feature vectors.

Here are the important bits:
* We first write the biases into the accumulator SIMD types. 
* We're iterating over 2 arrays, as we have a separate weight for each pair of neurons.
* We don't have to pad the array that's being used as input, just the one that will hold the output. This is because we're doing vectorization only over the output neurons, not the input neurons.
* Once again, we're relying on the FMA operation to do the heavy lifting.

As expected, using the implementation using AVX-512 intrics leads to an uptick in performance:
* Naive: 1.21 seconds
* AVX-512: 0.21 seconds

This time, we get a speedup of around 6x.

# Wrapping up
As noted earlier, it's worthwile to check if some BLAS library would have what you need before starting to write intrinsics yourself.
These libraries often use some combination of intrinsics and hand-written assembly to extract optimum performance for common operations, such as matrix multiplications.
If your implementation is going to compete with theirs, it won't be you coming out on top.

Of course, I'm not claiming that the implementations shown here are 100% optimal.
They're clearly not, we're not even aligning our memory before loads and stores.
However, the remaining optimizations wouldn't change the conclusions that we can already draw from these results.
Furthermore, the point of this post has been to showcase the general process of optimizing code using AVX-512 intrinsics and the types of speedups that one could expect from doing so.
As is evident, optimization using intrinsics can be an extremely powerful tool in your toolbox.

All code used in these experiments can be found this [Github repository](https://github.com/roopekj/cpu-simd)
