---
title: "Mobius反演（待完善）"
description: ""
date: 2026-07-26T14:49:57+08:00
lastmod: 2026-07-26T14:49:57+08:00
draft: true

categories:
  - Math
  - CP-Algorithm
tags:
  - Algorithm

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、写在前面

翻到了好久之前没写完的blog，但最近实在没什么网瘾~~（游戏太好玩了）~~，先暂存，以后想捡起来了再回来完善吧。

不知道分类是放算竞里面还是放数学里面，暂且两个都放吧（



## 一、莫比乌斯函数

### 1.1 定义

**莫比乌斯函数（Möbius 函数）**定义为
$$
\mu(n)=\left\{\begin{array}{ll}
1, & n=1, \\
0, & n \text { is divisible by a square }>1, \\
(-1)^{k}, & n \text { is the product of } k \text { distinct primes. }
\end{array}\right.
$$


### 1.2 莫比乌斯函数 μ(n) 是积性函数

**证明：**
$$
\begin{align}
& 假设 m、n 互素，只需证 \mu(mn) = \mu(m)\mu(n) \\
& 对于 m = 1 或 n = 1 的特殊情况显然成立 \\
& 对于 m 含平方因子 或 n 含平方因子 的特殊情况显然成立 \\
& 否则，\mu(mn) = (-1)^{s+t} = (-1)^s(-1)^t = \mu(m)\mu(n) \\
& 证毕
\end{align}
$$


### 1.3 莫比乌斯函数的和函数

莫比乌斯函数的和函数在整数 n 处的值 $ F(n) = \sum_{d|n} \mu(d) $，满足
$$
\sum_{d|n} \mu(d)=\left\{\begin{array}{ll}
1, & n=1, \\
0, & n>1. \\
\end{array}\right.
$$


**证明：**
$$
\begin{align}
& n = 1 时显然成立 \\
& 否则，因为 \mu(n) 是积性函数，所以其和函数也是积性函数 \\
& 则由 F(p^k) = \mu(1)+\mu(p)+\mu(p^2)+\dots +\mu(p^k) = 1 + (-1) + 0 + \dots + 0 = 0 \\
& 可得，F(n) = F(p_1^{a_1})F(p_2^{a_2})\dots F(p_t^{a_t}) = 0 \\
& 证毕
\end{align}
$$


### 1.4 筛法

一般选择**埃氏筛**或者**线性筛**。

**埃氏筛写法**：

```c++
using i8 = int8_t;
constexpr int N = 1E5;
i8 mu[N + 1];
mu[1] = 1;
for (int i = 1; i <= N; ++i) {
    for (int j = i + i; j <= N; j += i) {
        mu[j] -= mu[i];
    }
}
```



**线性筛写法**：

```c++
constexpr int N = 1E5;
i8 mu[N + 1];
bool notPrime[N + 1];
std::vector<int> primes;
mu[1] = 1;
for (int i = 2; i <= N; ++i) {
    if (!notPrime[i]) {
        primes.push_back(i);
        mu[i] = -1;
    }
    for (int j : primes) {
        if (i * j > N) break;
        notPrime[i * j] = true;
        if (i % j == 0) {
            mu[i * j] = 0;
            break;
        } else {
            mu[i * j] = -mu[i];
        }
    }
}
```



## 二、莫比乌斯反演

### 2.1 定义

若 f 是算术函数，F为 f 的和函数，对任意正整数 n 满足
$$
F(n) = \sum_{d|n} f(d)
$$
则对任意正整数 n：
$$
f(n) = \sum_{d | n} \mu(d) F(n/d)
$$
证明：
$$
\begin{align}
\sum_{d|n}\mu(d) &= \sum_{d|n}\mu(d) \sum_{e|(n/d)}f(e) \\
&= \sum_{d|n} \sum_{e|(n/d)} \mu(d)f(e) \\
&= \sum_{e|n} f(e) \sum_{d|(n/e)} \mu(d) \\
\end{align}
$$
由 1.3 可知，$\sum_{d|(n/e)} \mu(d)$ 只有在 n = e 时取1，其余时候取0

故
$$
\sum_{e|n} f(e) \sum_{d|(n/e)} \mu(d) = f(n) \cdot 1 = f(n)
$$
证毕



### 2.2 对偶形式

若 f 是算术函数，F为 f 的和函数，对任意正整数 n 满足
$$
F(n) = \sum_{d|n} f(d)
$$
则对任意正整数 n：
$$
f(n) = \sum_{n | d} \mu(d/n) F(d)
$$
证明方法类似，不再赘述。



## 三、例题

### 3.1 和函数为积性函数推原函数为积性函数

如题：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785055095756_image.png)

证明：
$$
\begin{align}
f(mn) &= \sum_{d|n} \mu(d) F(mn/d)	\\
&= \sum_{d_1|m, d_2|n} \mu(d_1) \mu(d_2) F(m/d_1) F(n/_d2) \\
&= \sum_{d_1|m} \mu(d_1) F(m/d_1) \sum_{d_2|n} \mu(d_2) F(n/d_2) \\
&= f(m)f(n)
\end{align}
$$

### 3.2 P2522 [HAOI2011] Problem b

>   从这个题学习求和式中 gcd 条件变形的技巧



**原题链接**

[P2522 [HAOI2011] Problem b](https://www.luogu.com.cn/problem/P2522)



**思路分析**

对要求的式子进行变形：

![66C2B64D8BA9537702A2EF8CD3ABECA4.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785063097008_66C2B64D8BA9537702A2EF8CD3ABECA4.png)

然后由于 n / d，m / d 是可以数论分块分段求的，那么我们只需要预处理 mu 的前缀和，求出每段的莫比乌斯函数和就好了。



**AC代码**

```c++
#include <bits/stdc++.h>
namespace ranges = std::ranges;
using i64 = long long;

constexpr int N = 5E4;
int mu[N + 1];

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    mu[1] = 1;
    for (int i = 1; i <= N; ++i) {
        for (int j = i + i; j <= N; j += i) {
            mu[j] -= mu[i];
        }
    }
    for (int i = 1; i <= N; ++i) mu[i] += mu[i - 1];

    auto get = [&](int a, int b, int k) -> i64 {
        i64 res = 0;
        a /= k; 
        b /= k;
        int top = std::min(a, b);

        for (int l = 1, r; l <= top; l = r + 1) {
            r = std::min({top, a / (a / l), b / (b / l)});
            res += 1LL * (a / l) * (b / l) * (mu[r] - mu[l - 1]);
        }
        return res;
    };

    int n;
    std::cin >> n;
    for (; n-- > 0; ) {
        int a, b, c, d, k;
        std::cin >> a >> b >> c >> d >> k;
        std::cout << get(b, d, k) - get(a - 1, d, k) - get(b, c - 1, k) + get(a - 1, c - 1, k) << '\n';
    }   

    return 0;
}
```



### 3.3 MC0485 刘姥姥的难题

**原题链接**

[MC0485刘姥姥的难题](https://www.matiji.net/exam/brushquestion/85/4693/305EE97B0D5E361DE6A28CD18C929AF0)

**思路分析**

![09506FE0FA8663C3DCE2640A0BC79A94.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785066433373_09506FE0FA8663C3DCE2640A0BC79A94.png)

>   bonus：将式子再进行一步推导，然后做到 O(nlnn) 预处理，O(1) 回答

**AC代码**

```cpp
#include <bits/stdc++.h>
using i64 = long long;
constexpr int P = 998244353;
const int N = 1E6;

i64 f[N], g[N];
int mu[N];

void init(int n) {
    f[1] = 1;
    for (int i = 2; i <= n; ++i) {
        f[i] = (f[i - 1] + f[i - 2]) % P;
    }
    mu[1] = 1;
    for (int i = 1; i <= n; ++i) {
        for (int j = i; j <= n; j += i) {
            ++g[j];
            if (j > i) {
                mu[j] -= mu[i];
            }
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;
    init(n); 

    i64 ans = 0;
    for (int d = 1; d <= n; ++d) {
        i64 val = 0;
        for (int e = 1; e * d <= n; ++e) {
            i64 c = n / (d * e);
            val += 1LL * c * c % P * c % P * mu[e] % P;
            val %= P;
        }
        ans += 1LL * g[d] * f[d] * val % P;
        ans %= P;
    }
    ans %= P;
    if (ans < 0) ans += P;
    std::cout << ans << '\n';

    return 0;
}
```















 
