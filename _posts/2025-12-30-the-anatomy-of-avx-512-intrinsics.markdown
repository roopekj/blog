---
layout: post
title:  "The anatomy of AVX-512 intrinsics"
date:   2025-12-30 00:00:00
categories:
---

$$\require{amsmath}$$

Modern x86 CPUs support a special class of instructions that can be grouped under the same umbrella term: [Same Instruction Many Data (SIMD)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
SIMD intructions are commands that make the CPU run operations on arrays of data instead of their individual scalars while utilizing large, purpose-built registers inside the CPU.
This way, the CPU is able to carry out the operations in any way it sees fit.
This is in contrast to having it jump between running an operation on two scalars and reading the same instruction again for the next two scalars.

When these instructions were just starting to gain traction the mid-to-late 90s with [MMX](https://en.wikipedia.org/wiki/MMX_(instruction_set)) and [SSE](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions), you would've had to write the assembly yourself to use them.
This is still the only 100% bulletproof way to get what you want, which is why projects like FFMPEG are still developed this way.
You can find the rest of their justifications [here](https://github.com/FFmpeg/asm-lessons/blob/main/lesson_01/index.md).
However, an alternative has existed for the rest of us since the early 2000s through support in compilers like the GCC, namely [intrinsic functions](https://en.wikipedia.org/wiki/Intrinsic_function).
These are functions in some higher-level programming language that have a special meaning to the compiler and, in our case, tell it to achieve something with SIMD instructions.
Intrinsics have made writing programs that utilize SIMD instructions much more approachable and are, for the most part, just as production-ready as writing the assembly yourself.
Ever since the original [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) instruction set in early 2010s, intrinsics have been available and fully supported as soon as the instructions themselves.

It is worth pointing out already at this point, that modern compilers may sometimes attempt to use SIMD instructions on their own, even without you having written a single intrinsic.
However, this is still quite fiddly, as I will show later in this post.
This is not to say that these attempts by the compiler aren't welcome - quite the opposite, and it can be really useful when this goes right; you can check out Matt Godbolt's [recent video](https://www.youtube.com/watch?v=d68x8TF7XJs) for more information.
However, this post will be about how to read and write intrinsics.
Then, you can be certain that your code will compile into a program that taps into these capabilities.

We will be focusing on [AVX-512](https://en.wikipedia.org/wiki/AVX-512) intrinsics, which make use of the 512-bit registers introduced on commodity hardware with [Skylake](https://en.wikipedia.org/wiki/Skylake_(microarchitecture)) in 2017.
Note that many things laid out here will be very similar (if not identical) in earlier SIMD intruction sets, such as SSE2 and AVX2.
The only consistent difference will be the sizes of the registers used, which will be either 128 or 256 bits in earlier instruction sets.
However, I have less familiarity with these, so I won't cover them here.

The ground truth information on all x86 SIMD intrinsics (including these AVX-512 variants) can be found on [this website](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).
It lists every intrinsic in a neatly indexed manner.
However, there are a total of 5160 AVX-512 intrinsics, so trying to remember them all by heart is pointless.
Furthermore, having to look up every intrinsic you come across or that you want to utilize is slow, so it's a good idea to get a grasp of their general rules so you can figure things out by youself.
I will try to outline the mental model I've found useful when doing so in this post.

#### The SIMD types
Before we get to the intrinsics template, let's briefly talk about typing.
AVX-512 recognizes four SIMD types, which are then used to add typing to values that could be thought of as residing in some SIMD register.
They are as follows:
* `__m512h`: array of 32 individual 16-bit floats.
* `__m512`: array of 16 individual 32-bit floats. Note that there is no s-suffix.
* `__m512d`: array of 8 individual 64-bit floats.
* `__m512i`: array of some number of integers, depending on how many fit into 512 bits.

These are central to how you write code that utilizes intrinsics, so they are important to remember.
This naming convention is (as far as I know) consistent between different languages and libraries, so it shouldn't matter if you're using C/C++ with the `immintrin.h` header, Rust with the `core::arch::x86_64` module, or any other viable alternative.
Make note of the two-underscore prefix and the use of *m* instead of *mm*.
This means that you're dealing with a SIMD type.
Intrinsics themselves use *mm* and are prefixed with just one underscore.

There are a few additional types that are related to masks, but those are not relevant for now and will be covered later on in the post.

# The general template
Despite their confusing looks, all AVX-512 intrinsics adhere (roughly) to this shared template:  
`_mm512[_optional mask]_<operation>_<data type>`

The first part just defines that we are operating on 512-bit registers, as opposed to 128-bit registers (`_mm128`) or 256-bit registers (`_mm256`).
Let's go through the other parts in my subjective order of importance.

#### Operation
The operation just answers the question "What are we actually doing?".
The list of operations available includes all the things you'd imagine, like `add`, `sub`, `mul`, `div`, `max`, `min` and `abs`.
It's important to remember, however, that we are always working on arrays of data and not scalar values.
An operation like `add` is going to do element-wise addition on two arrays, and write the resulting array into some memory address.
You need to both formulate your input data as arrays and handle arrays as your output values.

You will also find some of these operations with a s-suffix, like `adds`, `subs`...
This stands for *saturation* and means that in case of overflow/underflow, the result is set to the datatype's maximum/minimum value instead of wrapping around.
Furthermore, operations can also have a variant that adds explicit rounding.
This can mean rounding to nearest integer, up/down an integer, truncation or following globally defined RC rules, located in the [MXCSR register](https://help.totalview.io/classicTV/current/HTML/index.html#page/Reference_Guide/Intelx86MXSCRRegister_2.html).
These intrinsics have an underscore between the base operation and the *round* modifier, as seen for instance in `_mm512_mul_round_ps` and `_mm512_add_round_ps`.
I would argue that despite the underscore, it's easiest to just see these as a new operation (eg. `mul_round`) instead of the normal operation (eg. `mul`) with an optional modifier.
Otherwise it makes the template more complicated for no good reason.

Bitwise logic operations are of course also available: `and`, `or`, `not`, `xor`...
They do exactly what you'd imagine they would.

To run any of these manipulations on data, you need to get said data into a SIMD register.
There are a few main ways to declare that some data should be delivered into a SIMD register.
These operations also construct and return a value of the appropriate SIMD type (`__m512h`, `__m512`, `__m512d` or `__m512i`), which can then be used as an operand in intrinsics that manipulate data.
The operations differ in their arguments:
* `load`: Takes a pointer as its argument and reads 512 bits starting from there. This means that data is always read from memory (CPU caches, RAM...).
* `set`: Takes as arguments the required number of scalar values to fill a SIMD register (ie. 16 singles, 8 doubles...). 
* `broadcast`: Takes a SIMD type of some smaller size (`__m128`, `__m256`) as its argument and broadcasts it in some way to this larger register. There are many variants of this with different suffixes.
* `set1`, `set4`: Takes as argument either one or four scalar values. The scalar or scalars are repeated across the SIMD register. A variant of the latter with an r-suffix does so in reversed order.
* `setzero`: Takes no arguments, zeroes out a register. This is especially useful when setting up accumulator registers.

For all operations except the first one, how the values end up in the register is left for the compiler to decide: memory access, register move, immediates...

The operation `fmadd` deserves a special mention here.
It stands for Fused Multiply-Add, and is the backbone of many linear algebra operations.
With `fmadd`, we are able to both multiply two arrays element-wise and sum their product with a third array in **0.5 clock cycles**.
Unless you're working with complex numbers, in which case this takes a full clock cycle.

Once data has been operated on in a SIMD register, we usually want to write it back into memory.
This is done with the `store` operation.
Intrinsics using `store` as their operation take as arguments a SIMD type and an arbitrary second pointer, and then write the contents of the former into the latter in-place.

This is a good point to wrap up the section on operations.
In summary, the operations are sort of what you'd expect, but there are many of them to remember. 
The complete list of operations of course just goes on and on: variants that only store the high/low bits of the result, steps from cryptographic algorithms like AES, trigonometric functions...
However, most of the programs being written with AVX-512 intrinsics probably gets done with just the operations listed here alone, and the rest you can look up as you go.

#### The curious detail about unaligned memory accesses
After explaining these operations, one has to mention `loadu` and `storeu`.
They're the same as their non-u counterparts, except that they don't require that the pointer being used for reading/writing is aligned.
If you know what this means, you can skip the following paragraph.
If not, keep reading.

Data is moved from RAM to the CPU cores' internal (L1, L2) or shared (L3) caches in units called *cache lines*.
These are just chunks of data pertaining to some static size depending on your CPU architecture.
You can think of this like how data is moved between RAM and disks in units called *pages*, all containing data whose size equals the *page size* (often 4KiB).
The comparison is not quite accurate as pages are not a hardware consideration but an abstraction left to the operating system, but the mental model is similar.
To make this point clear, reading data from RAM to caches is not done in individual bits or bytes.
Instead, the CPU will fetch data from RAM one cache line at a time; it doesn't know an operation that would fetch "half a cache line".
In this way, the cache line is the smallest unit for data transfer to and from caches, as seen by the cache subsystem of the CPU.
The size of a cache line is typically 64 bytes (512 bits), which means that one of our 512-bit registers can get its data from just one cache line.
However, if the 512 bytes of data you want to move to the cache is not neatly stored in one cache line, but instead crosses the boundary between two cache lines, the CPU will have to fetch both to get all the data.
In addition, the CPU will have to merge the contents of the two cache lines in order to form the 512 bits that can actually be delivered to the cache, from where they will then be moved to a register.
This causes a slowdown every time we get a cache miss, as the CPU will have to fetch twice as much data using twice as many memory accesses, and then perform some additional work.

To safely use `load` and `store`, you need to have aligned the data being pointed at along the cache lines.
Otherwise, this will cause undefined behavior.
For a simple array of 16 floats on the stack, this would mean defining the variable like this in your code: `alignas(64) float arr[16]`.
For data on the heap, you could use `std::aligned_alloc`.
In any case, having done the alignment, the pointer points to a (virtual) memory address that is divisible by 64.
Therefore, as long as you're reading/writing 512 bits and your cache line size is in fact 64 bytes, this can be done in one memory access that reads/writes a single cache line, meaning that `load` and `store` will work as expected.

On the other hand, `loadu` and `storeu` will instead just obediently do all the extra work if the data is not aligned.
As per my understanding, using these operations with data that **is** in fact aligned does not cause a slowdown on modern CPUs.
However, they could end up providing a computationally costly band-aid solution to bad data alignment that you really don't want to rely on.

There are many more considerations lurking here that, although important, are way above by paygrade.
Things like prefetching, out-of-order accesses and how modern CPUs start doing speculative loads of adjacent cache lines as soon as you even think about a specific memory address.
However, as a general rule, you should have your frequently used data aligned at allocation time.
The only real danger is if you have a massive number of small objects that are aligned via `alignas` on the type itself, as this could end up wasting a lot of memory.
For instance, this code:
{% highlight c++ %}
struct alignas(64) OneByteOfData {
    uint8_t x;
};
OneByteOfData arr[10];
{% endhighlight %}
Would end up using 640 bytes worth of memory even though we only have 10 bytes of data.
This is because the C++ standard defines that `sizeof(T)` must be a multiple of `alignof(T)`.
Even in this (highly dubious) code, you could add the alignment on the array instead of the struct itself and you'd be fine, so early aligning is a good thing to get in the habit of doing.
Also, if you want to run your program on M3/M4 macs, the cache line size is actually 128 bytes.
Have fun.

#### Data type
Learning the data types of AVX intrinsics is sort of like remembering register names is assembly. 
Once you brute-force the process of internalizing them, they can become second nature.
Until then (and after you forget them two days later), they can be a real headache.

As a forewarning, the "e" originally comes from the term "extended packed integer" but it doesn't actually fit into the modern naming convention and is probably easiest to just ignore.
You will see it with integer data types, but it's just a historical artifact from the hayday of SSE2 where they needed to distinguish from older integer operations.

With that out of the way, here are the common data types:
* `ph`: **P**acked **h**alf, a 16-bit floating point number
* `ps`: **P**acked **s**ingle, a 32-bit floating point number
* `pd`: **P**acked **d**ouble, a 64-bit floating point number
* `epi[8,16,32,64]`: **P**acked **i**nteger, a signed integer of some size
* `epu[8,16,32,64]`: **P**acked **u**nsigned integer, an unsigned integer of some size
* `si512`: **S**igned **i**nteger, 512 bits of signed integer data

The last data type is available when storing/loading data and a few other operations that don't care about the specific data types.

#### Data types when using load/store operations
Now is a good moment to talk about how these data types are used together with load/store operations.
If you're loading or storing floating point numbers, things go as you'd expect.
You use operations like `_mm512_load_ps` to load 32-bit floats into a register, then write them back into memory with `_mm512_store_ps`.
These intrinsics exist for all three floating point types, with both an aligned (`load`/`store`) and an unaligned (`loadu`/`storeu`) variant being available.
However, things are not as simple with integers.

You may notice that there is no `ui512` data type, ie. a data type that would hold 512 bits worth of unsigned integers.
This is because you're supposed to use `si512` for unsigned integer data as well.
Yes, you can (and should) use `_mm512_load_si512` for both signed **and** unsigned integers.
The logic is that the operation is just copying bits from one place to another, so it's going to work regardless of what's being read; there's no arithmetic going on here.
So, I guess it doesn't really matter.

For completeness, here are the intrinsics available for loading/storing integers of a specific size:
{% highlight c++ %}
// Unaligned load   Unaligned store
_mm512_loadu_epi8   _mm512_storeu_epi8
_mm512_loadu_epi16  _mm512_storeu_epi16
_mm512_loadu_epi32  _mm512_storeu_epi32
_mm512_loadu_epi64  _mm512_storeu_epi64

// Aligned load     Aligned store
_mm512_load_epi32   _mm512_store_epi32
_mm512_load_epi64   _mm512_store_epi64
{% endhighlight %}

Two things worth noting:
* There are **no** variants of store/load intrinsics whose data type is an unsigned integer. You're always supposed to use an intrinsic that is meant for signed integers. So instead of `_mm512_loadu_epu8`, which doesn't exist, you would use `_mm512_loadu_epi8`.
* There are **no** store/load intrinsics that enforce memory alignment for 8-bit or 16-bit integers. Don't ask me why, I don't know.

Furthermore, you could replace any one of these intrinsics and more with a `si512` variant, as those exist for both aligned and unaligned loads and stores:
{% highlight c++ %}
_mm512_load_si512   _mm512_store_si512
_mm512_loadu_si512  _mm512_storeu_si512
{% endhighlight %}

I suppose there may be some benefit in being explicit about how many individual elements your register holds.
Definitely not to the compiler - you don't need to tell it and it certainly doesn't care if you do.
The compiler treats all loads as just writing 512 bits from memory to a 512-bit register, including the ones meant for floating point numbers.
The same goes for stores but with the direction reversed.

Also, any intrinsic that performs arithmetic on the loaded data will always specify the exact data type.
So, loading with the data type is not useful there either.
Maybe in terms of code completion in the editor?
Readability?
If so, then why are there no 8-bit or 16-bit variants or aligned loads or stores?

Again, don't ask me why, I don't know.
Smells like design by committee to me.
In any case, I usually just use the `si512` variants and couldn't tell you why this would be a bad idea.
I'm not above being wrong but I've decided to find out the hard way.
Therefore, I can't recommend you do anything except the same.

#### Mask
Some operations can benefit from being masked.
A masked operation means that it's possible for the operation to only affect a subset of the operands' individual elements.
Masking is activated by the optional *mask* part of the template, just before the operation.
There's two variants of this: `mask` and `maskz`.
So for instance, you can use either `_mm512_mask_add_epi64` or `_mm512_maskz_add_epi64`.
These two differ in what happens when the mask has disabled the operation for one of the elements.

* In `mask`, the element is then copied from the first input register. This is sometimes known as *merge masking*.
* In `maskz`, the element is set to 0. This is sometimes known as *zero masking*, hence the z-suffix.

The usefulness of masks, in my experience, has been split into two pronounced categories.

First, in operations such as masked self-attention or when using padded inputs with an encryption algorithm, some of the input elements should be ignored. You can achieve this with a mask that defines which elements are to be left out.

Here's a short code snippet demonstrating this:
{% highlight c++ %}
// Some array of data
int arr[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

// The first three elements of the data are meaningful, the rest 
// should be zeroed out.
__mmask16 mask = 0b1110000000000000;

// The register will hold the values {1,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0}
__m512i data = _mm512_maskz_loadu_epi32(mask, arr);

{% endhighlight %}

Second, tail handling. If you're running an operation on an array whose size is not divisible by 512 bits, you need to do tail handling. This is to say, the last bits that don't fill an entire register need to be processed in some way. Masking can be used here to ignore the high bits that are not actually part of the relevant data.

Again, here's a short code snippet.
This is taken from a function that adds a constant value to all elements in a very long array of unsigned 8-bit integers:
{% highlight c++ %}
__m512i addV = _mm512_set1_epi8((signed char)valueToAdd);

/* (loop over data in 512-bit increments) */

// 'total' is how many elements there are in the array, 'i' is the index
// from which incrementing by 512 bits would've gone over the array's size
size_t remaining = total - i;

if (remaining > 0) {
  // For example, with remaining == 2
  // 0000 0001 -> 0000 0100 -> 0000 0011
  __mmask64 mask = (1ULL << remaining) - 1;

  // Load the data into a register with the mask, other elements are zeroed
  // out. Note that we have to load with the wrong data type.
  __m512i tail = _mm512_maskz_loadu_epi8(mask, &data[i]);

  // Calculate the saturated add on the data.
  __m512i res = _mm512_adds_epu8(tail, addV);

  // Write the result from the register back into the data array using the
  // same mask, so that we don't overwrite prior values with valueToAdd.
  _mm512_mask_storeu_epi8(&data[i], mask, res);
}
{% endhighlight %}

As you've perhaps noticed, masking adds a few more SIMD types, such as the `__mmask16` and `__mmask64` above.
These are just bit sequences with a specific size, where the value 0 indicates an element that is masked out, and the value 1 indicates an element that is active.
As masking is done element-wise, the different sizes pertain to the number of elements that can be present in a register, ie. 64 individual 8-bit integers, 8 individual 64-bit floats, and so on.
The types themselves are nothing too complex; They're just instances of some ordinary numeric type that the compiler treats in a special way when used in intrinsics:
{% highlight c++ %}
typedef unsigned char __mmask8;
typedef unsigned short __mmask16;
typedef unsigned int __mmask32;
typedef unsigned long long __mmask64;
{% endhighlight %}

I'm sure there are a million more ways to make use of masks, but these are the ones that I have found useful.
Feel free to do more digging on your own, there is an endless supply of intrinsics where you can just add masking to perform the same operation a bit differently.

# Things that are missing in AVX-512
It would be reasonable to assume that anything you'd imagine is in the list of available intrinsics.
You won't find anything too obvious missing, but some more niche things have definitely been left out.
For instance, there are no `_mm512_add_epu[8,16,32,64]` intrinsics.
This is to say, you can't add together two arrays of unsigned integers with wraparound.
The saturated variants are available (`_mm512_adds_epu[8,16,32,64]`), as are the variants for signed integers (`_mm512_add_epi[8,16,32,64]`), but the unsigned and non-saturated versions are not.

Things like this come down to the required effort from the chip manufacturers vs. perceived usefulness.
This operation just isn't common enough to warrant either the hardware support on 512-bit registers or the developer time on implementing the intrinsic, plain and simple.
Of course, you do have options if you really need something like this.
You can create some Frankenstein concoction from unrelated intrinsics, or just use AVX2/SSE2 instead (both of which provide such an intrinsic).
Whenever you find that your specific needs are not supported, it's also a good moment to think about what you're trying to achieve.
Sometimes there's a reason why a specific way of doing things has been deemed unnecessary.

# That's all there is to it
We have now covered nearly everything I wanted to say about AVX-512 intrinsics.
There's really nothing too insiduous hiding in how they're written.
Sure, they carry a bit of historical baggage, but so does everything that's sufficiently close to hardware.
As an exercise to see how well the template has sunk in so far, here's a small exam for you to test your undestanding.
Try to figure out what these common intrinsics do.
What are they doing?
With what data type?
How many elements are they operating on at the same time?
There are solutions in the form of plain english explanations at the bottom of the post.

{% highlight c++ %}
_mm512_loadu_ps
{% endhighlight %}

{% highlight c++ %}
_mm512_set1_epi8;
{% endhighlight %}

{% highlight c++ %}
_mm512_fmadd_ph
{% endhighlight %}

{% highlight c++ %}
_mm512_adds_epu8
{% endhighlight %}

There's just one extra point of possible confusion that I want to clarify before wrapping up this post.


# What do our intrinsics compile into?
When first learning about intrinsics, one may assume that they compile directly to assembly in some deterministic way.
After all, they are quite low level, as we are defining operations that target specialized registers.
While there's some truth to this, one must understand that this is in no way guaranteed.
For instance, the compiled version of this program

{% highlight c++ %}
#include <immintrin.h>
__m512i a = _mm512_set1_epi32(5);
__m512i b = _mm512_add_epi32(a, a);
{% endhighlight %}

might contain something like this  

{% highlight nasm %}
mov     eax, 5
vpbroadcastd zmm0, eax
vpaddd       zmm1, zmm0, zmm0
{% endhighlight %}

which would mean a direct one-to-one mapping. The compiled version might also look nothing like this.

If the compiler thinks this is not efficient in the current context, it will do something different.
The compiler might also produce a binary that spills this value onto the stack at some point because it runs out of registers.
You really don't know unless you analyze the compiled program.

For instance, the above three-line program actually compiles into this with some optimizations enabled:

{% highlight nasm %}
_GLOBAL__sub_I_a:
        mov     eax, 5
        mov     edx, 10
        vpbroadcastd    zmm0, eax
        vpbroadcastd    zmm1, edx
        vmovdqa64       ZMMWORD PTR a[rip], zmm0
        vmovdqa64       ZMMWORD PTR b[rip], zmm1
        vzeroupper
        ret
b:
        .zero   64
a:
        .zero   64
{% endhighlight %}

It completely skips the addition and just writes the resulting value of 10 into the register without calculating it.
Neither register is used for any operations during runtime.

The point is that when we receive a value of some SIMD type (`__m512i` in this case), it isn't a pointer to a register being used to store data or some other low-level construct.
It's just like any other regular type, just an abstraction of what sort of data you will get when you read the value.
In this case, it would be 16 signed integers with a size of 32 bits each, totaling 512 bits of ordinary binary data.
For an uint8_t value it would be 8 bits depicting an unsigned integer.
If you want to be pedantic, you can call it an instance of a [opaque data type](https://en.wikipedia.org/wiki/Opaque_data_type).
Just because you have a variable of that type available to you, doesn't mean that you are defining where the data is or what the compiled binary will do with it.
The data may be in a register, the stack, the heap, or even on disk if things are really bad.
It all comes down to how the compiler decides to use your CPU, and how the OS manages your system resources during execution with regards to swapping, interrupts and so on.

In fact, if you look at the headers that define these types, you will find definitions like this:

{% highlight c++ %}
typedef float __m512 __attribute__((__vector_size__(64), __aligned__(64)));
{% endhighlight %}

The type itself is just a vector with some special codegen rules that the developer (thankfully) does not need to care about.
For those who wish to care anyways, it's a type based on a float, but whose size is defined as being 64 bytes and whose alignment will be along the cache lines.
This turns it into a vector with 16 floats, with a size of 512 bits, that is aligned on a 64-byte boundary from the compiler's point of view.
Don't get me wrong - you can (and I personally do) use these values as if they were registers.
Still, it's good to keep in mind the inaccuracy of that mental model in case you ever run into performance issues that can't be explained by it.

# Final thoughts
SIMD instructions are a neat way to speed up your programs, and they are relatively approachable thanks to intrinsics.
It's worth mentioning that you shouldn't rule out older SIMD instruction sets like AVX2 and SSE2, as they are definitely still useful.

First of all, AVX-512 is still not supported on much of the hardware used today. 
CPUs made by Intel have had support since 2017 and CPUs made by AMD have had half-assed support since late 2022.
Real support on AMD CPUs only arrived in 2024.
This is in contrast to AVX2 (supported on all commodity CPUs since 2013) and SSE2 (supported on literally every x86-64 CPU ever made).

Secondly, older instruction sets can actually turn out to be equally fast or even faster than AVX-512 in some situations.
The high power draw (and consequent heat output) of using AVX-512 can make the CPU underclock aggressively, among other oddities in how the latencies differ among the instruction sets.

Third, AVX-512 is once again **unsupported** on new intel CPUs of the [Alder Lake](https://en.wikipedia.org/wiki/Alder_Lake) family, as the instructions are not supported on the new *efficiency cores* that they just had to push out half-baked.
There's seemingly a new instruction set in the works that promises to fix this.
It's called [AVX10](file:///home/rka/Downloads/355989-intel-avx10-spec.pdf) in Intel's specification, and it's being touted as this grand new instruction set that unifies all the previous ones.
I've heard nothing but bad things said about it, but only time will tell how things will turn out.

Fourth and finally, if you immediately agree with everything Linus Torvals has to say, you should be hoping that AVX-512 [dies a painful death](https://www.zdnet.com/article/linus-torvalds-i-hope-intels-avx-512-dies-a-painful-death/) and Intel completely stops optimizing floating point arithmetic.
I can't help you with this one.

The point I'm trying to make is this: Don't assume that AVX-512 should necessarily be your go-to instruction set even though it has the largest registers.
In fact, if you need universal hardware support, then you have to rule out everything except SSE2.
Otherwise, the choice still isn't an exact science.
Thankfully, when writing your code with intrinsics, experimenting is not such an arduous process.
Once you've written a bit of code with intrinsics, you will also come to realize that they do in fact follow a template that is surprisingly simple.


#### Exam solutions

* `_mm512_loadu_ps`: Read 16 individual 32-bit floating point numbers from a possibly unaligned pointer, then write them into a SIMD register
* `_mm512_set1_epi8`: Takes a 8-bit signed integer as its argument, and fills a SIMD register with replicas of it.
* `_mm512_fmadd_ph`: Takes three `__m512h` values as arguments, each containing 32 individual 16-bit floating point numbers, and sums the element-wise product of the first two with the third.
* `_mm512_adds_epu8`: Takes two `__m512i` values as arguments, both containing 64 individual 8-bit unsigned integers, and sums them element-wise. If any element overflows, its value is set to 255.
