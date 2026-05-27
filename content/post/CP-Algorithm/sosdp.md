---
title: "SOS DP"
description: ""
date: 2026-05-26T21:10:34+08:00
lastmod: 2026-05-26T21:10:34+08:00
draft: false

categories:
  - CP-Algorithm
tags:
  - Dynamic Programming
  - bitmask
  - set theory
  - Algebra

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、写在前面

随便写写。子集和超集和的计算就是高维前后缀和，子集反演超集反演的计算就是高维前后缀差分。还是比较easy的。



## 一、SOS DP

### 1.1 高维前缀和

**SOS DP (Sum Over Subsets Dynamic Programming)**，也被称为**高维前缀和**，是算法竞赛中处理位运算（尤其是子集、超集问题）的一项极为优雅和高效的技巧。

给定一个大小为 $2^n$ 的数组 $A$（下标从 $0$ 到 $2^n-1$），我们需要计算一个新数组 $F$，使得：
$$
F[mask] = \sum_{i \subseteq mask} A[i]
$$
（注：$i \subseteq mask$ 表示 $i$ 是 $mask$ 的子集，即 `(i & mask) == i`）



**1. 暴力做法 $O(3^n)$**

最直观的做法是对于每个 $mask$，枚举它的所有子集：

```c++
for (int mask = 0; mask < (1 << n); ++mask) {
    F[mask] = A[0];
    for (int i = mask; i > 0; i = (i - 1) & mask) { // 经典枚举子集位运算技巧
        F[mask] += A[i];
    }
}
```



**3. 高维前缀和 $O(n \cdot 2^n)$**

思考初学算法的时候，怎么求一维数组前缀和？——从左往右扫一遍。

怎么求二维数组前缀和（不用一次遍历的容斥写法）？——先对每一行做一维前缀和，再对每一列做一维前缀和。

**扩展到n维**——依次对第 $0$ 维、第 $1$ 维、...、第 $n-1$ 维做一维前缀和。

**状态定义**：

$dp[i][mask]$ 表示在只允许改变 $mask$ 的前 $i$ 位（即第 $0$ 到第 $i-1$ 位）的前提下，所有子集的和。

**状态转移**：
考虑 $mask$ 的第 $i$ 位：
* 如果 $mask$ 的第 $i$ 位是 `0`：它的子集在这一位也必须是 `0`，所以 $dp[i][mask] = dp[i-1][mask]$。
* 如果 $mask$ 的第 $i$ 位是 `1`：它的子集在这一位可以是 `0` 也可以是 `1`。
  因此 $dp[i][mask] = dp[i-1][mask] + dp[i-1][mask \oplus (1 \ll i)]$。

**空间优化（滚动数组）**：
因为 $dp[i]$ 只依赖于 $dp[i-1]$，我们可以省去第一维：

```cpp
for(int i = 0; i < n; ++i) { // 枚举维度
    for(int mask = 0; mask < (1 << n); ++mask) { // 枚举所有状态
        if(mask & (1 << i)) { // 如果第 i 位是 1
            F[mask] += F[mask ^ (1 << i)]; // 加上第 i 位为 0 的状态
        }
    }
}
```


### 1.2 存在性与最值问题

SOS DP 不仅仅能求和，只要满足**结合律**和**交换律**的操作（如 max⁡,min⁡，按位或/与），都可以用 SOS DP。

一个经典问题：[E. Compatible Numbers](https://codeforces.com/problemset/problem/165/E)

对数组中每个数A[i] 找到 一个 A[j] 使得 A[i] & A[j] = 0。

我们利用 sosdp 求子集max，然后每个数的答案就是 dp[~A[i] & U]

**代码实现(C#)：**

```C#
public void Solve() {
    int n = br.ReadInt32();
    int[] a = br.ReadInt32(n);
    int U = 1 << (int.Log2(a.Max()) + 1);
    int[] dp = new int[U];
    foreach (var x in a) {
        dp[x] = x;
    }
    for (int i = 0; i < 30; ++i) {
        for (int s = 1; s < U; ++s) {
            if ((s >> i & 1) > 0) {
                dp[s] = Math.Max(dp[s], dp[s ^ (1 << i)]);
            }
        }
    }
    bw.AppendJoin(" ", a.Select(x => ~x & (U - 1)).Select(x => dp[x] > 0 ? dp[x] : -1));
}
```



### 1.3 求超集

**题意**：不求子集了，求超集。即 $F[mask] = \sum_{mask \subseteq i} A[i]$。

这个很好求，就是把**高维前缀和**变成**高维后缀和**。

```c++
for(int i = 0; i < n; ++i) {
    for(int mask = (1 << n) - 1; mask >= 0; --mask) { // 从小到大也可以
        if(!(mask & (1 << i))) { // 重点：如果第 i 位是 0
            F[mask] += F[mask ^ (1 << i)]; // 加上第 i 位为 1 的超集状态
        }
    }
}
```



### 1.4 高维差分

前/后 缀和的逆运算是差分，那么高维前/后缀和 的逆运算就是高维差分。

在很多场景中，利用高位前缀和做高维差分，被称为**子集反演**。利用高维后缀和做高维差分，被称为**超集反演**。

以子集反演为例，子集反演的过程是**从“至多”到“恰好”**：

在组合数学中，我们经常会遇到两个函数 $f(S)$ 和 $g(S)$，其中 $S$ 是一个集合。

假设 $g(S)$ 表示“**恰好**是集合 $S$”的值，而 $f(S)$ 表示“**包含于**集合 $S$ 的所有子集”的值之和（即“至多”是 $S$）。
它们的关系是：
$$f(S) = \sum_{T \subseteq S} g(T)$$
如果我们已知 $f$ 数组，想要反推 $g$ 数组，这就需要用到**子集反演公式**：
$$g(S) = \sum_{T \subseteq S} (-1)^{|S| - |T|} f(T)$$
*(其中 $|S|$ 表示集合 $S$ 的大小，即二进制中 1 的个数)*。

>   我们发现，和S 相差元素为奇数，那么贡献是负的，偶数，贡献是正的，这其实就是容斥原理的应用。

代码实现很简单：

```c++
// 初始时 F 数组里存的是 f(S)
for(int i = 0; i < n; ++i) {
    for(int mask = 0; mask < (1 << n); ++mask) {
        if(mask & (1 << i)) { // 第 i 位是 1
            F[mask] -= F[mask ^ (1 << i)]; // 减去第 i 位是 0 的情况
        }
    }
}
// 结束时 F 数组里存的就是 g(S)
```

>   Q：为什么代码实现中一律全是减号？
>
>   A：
>
>   sosdp 就是沿着dag求和的过程，在逐层做减法的过程中自动完成了奇偶交替符号的容斥计算。



一个板题：[D. Jzzhu and Numbers](https://codeforces.com/contest/449/problem/D)

计算有多少个子集满足按位与为0，这显然需要我们做超集反演。

**代码实现（C#）：**

其中，Z是取模数

```C#
public void Solve() {
    int n = br.ReadInt32();
    int[] a = br.ReadInt32(n);
    int hi = int.Log2(a.Max()) + 1;
    int U = 1 << hi;
    Z[] dp = new Z[U];
    foreach (var x in a) {
        ++dp[x];
    }

    for (int i = 0; i < hi; ++i) {
        for (int s = 0; s < U; ++s) {
            if ((s >> i & 1) == 0) {
                dp[s] += dp[s ^ (1 << i)];
            }
        }
    }

    for (int s = 0; s < U; ++s) {
        dp[s] = ((Z)2).Pow(dp[s].Value) - 1;
    }

    for (int i = 0; i < hi; ++i) {
        for (int s = 0; s < U; ++s) {
            if ((s >> i & 1) == 0) {
                dp[s] -= dp[s ^ (1 << i)];
            }
        }
    }

    bw.AppendLine(dp[0].Value);
}
```

