---
toc: true
layout: post
description: 선형축소방법론
categories: [MDS, Dimensionality reduction]
title: Multidimensional Scaling(MDS, 다차원 척도법)
---

# Multidimensional Scaling(MDS, 다차원 척도법)

선형 변환의 차원 축소 기법인 동시에 변수 추출 기법인 다차원 척도법에 대해서 설명하겠습니다.

다차원 척도법은 $n$개의 개별적인 객체들 사이의 유사도 또는 비유사도를 측정하여 얻어진 distance matrix를 이용하여 고차원의 공간에서 존재하는 객체 간의 거리(유사도) 정보가 저차원의 공간에서도 최대한 보존되는 좌표계(coordinate)를 찾는 것을 목표로 합니다.

간단한 Distance matrix $\mathbf{D}$를 예로 들자면 아래와 같은 행렬처럼 나타낼 수 있습니다.

![Multidimensional%20Scaling(MDS,%20%E1%84%83%E1%85%A1%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%20%E1%84%8E%E1%85%A5%E1%86%A8%E1%84%83%E1%85%A9%E1%84%87%E1%85%A5%E1%86%B8)%20e38e1a4bbcb94af2ae415bb3c508d4c1/Untitled.png](<Multidimensional%20Scaling(MDS,%20%E1%84%83%E1%85%A1%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%20%E1%84%8E%E1%85%A5%E1%86%A8%E1%84%83%E1%85%A9%E1%84%87%E1%85%A5%E1%86%B8)%20e38e1a4bbcb94af2ae415bb3c508d4c1/Untitled.png>)

우리가 흔히 접하는 일반적인 데이터 $\mathbf{X}$를 이용하여 distance matrix $\mathbf{D}$를 표현하는 것은 쉽습니다(Cosine, Jaccard, Euclidean, Manhattan 등을 이용하여 $\mathbf{X} \Rightarrow \mathbf{D}$). 하지만 distance matrix $\mathbf{D}$를 원 데이터 $\mathbf{X}$로 나타내는 것은 불가능합니다. 왜냐하면 위 행렬을 예시로 든다면 사과를 임의의 차원의 벡터로 표현할 수 있는 방법이 없기 때문입니다. 따라서 또 다른 선형 변환 차원 축소법인 동시에 변수 추출법인 PCA보다 MDS의 범용성이 훨씬 높습니다. 즉, PCA를 적용할 수 있는 모든 데이터는 MDS를 적용 가능하지만, MDS를 적용할 수 있는 모든 데이터에 대해서는 PCA를 적용할 수 없습니다.

구체적인 MDS의 절차는 다음과 같습니다.

**우선, 데이터가 거리(유사도) 정보로 이루어져 있지 않다면 우선 거리(유사도) 행렬로 만들어줍니다.** 만약에 객체들 사이의 좌표값들이 존재할 경우 그 객체간의 거리(유사도) 를 계산할 수 있습니다.

거리 행렬이 가져야하는 특징은

(1) $d_{ij} \geq 0$, 임의의 두 객체 사이의 거리는 0보다 크거나 같아야 한다.
(2) $d_{ii} = 0$, 나와 나 사이의 거리는 0이어햐 한다.
(3) $d_{ij}=d_{ji}$, 객체 $i$와 객체 $j$ 사이의 거리는 같으므로 $d_{ij}=d_{ji}$인 symmetric matrix이아야 한다.
(1), (2), (3)에 추가적으로 거리는 삼각 부등식이라 불리는 $d_{ij} \leq d_{ik}+d_{jk}$를 만족해야 한다.

![Multidimensional%20Scaling(MDS,%20%E1%84%83%E1%85%A1%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%20%E1%84%8E%E1%85%A5%E1%86%A8%E1%84%83%E1%85%A9%E1%84%87%E1%85%A5%E1%86%B8)%20e38e1a4bbcb94af2ae415bb3c508d4c1/Untitled%201.png](<Multidimensional%20Scaling(MDS,%20%E1%84%83%E1%85%A1%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%20%E1%84%8E%E1%85%A5%E1%86%A8%E1%84%83%E1%85%A9%E1%84%87%E1%85%A5%E1%86%B8)%20e38e1a4bbcb94af2ae415bb3c508d4c1/Untitled%201.png>)

원 데이터 $\mathbf{X}$ matrix($d \times n$)이고 이를 $\mathbf{D}$ matrix($n \times n$)로 만듭니다. $\mathbf{D}$는 upper triangle과 lower triangle이 서로 대칭입니다.

**둘째, 거리 정보를 보존하는 좌표계를 찾습니다.**

위에서 생선한 $\mathbf{D}$ matrix에서 index를 $r$과 $s$로 하는 임의의 객체에 대해서

$d_{rs}^2 = (\mathbf{x}_{r}-\mathbf{x}_{s})^T(\mathbf{x}_{r}-\mathbf{x}_{s})$ 로 나타낼 수 있습니다. 하지만 $\mathbf{x}$에 대한 정보를 다이렉트로 찾기가 어렵기 때문에 $\mathbf{B}$라는 inner product로 만들어진 $n \times n$ 매개 행렬을 이용할 겁니다. $\mathbf{B}$ matrix에서 index를 $r$과 $s$로 하는 임의의 객체는 $b_{rs} = \mathbf{x}_{r}^T\mathbf{x}_{s}$로 벡터의 내적으로 표현합니다. 즉, $\mathbf{D}_{n\times n} \Rightarrow \mathbf{B}_{n \times n} \Rightarrow \mathbf{X}_{d \times n}$의 흐름을 따릅니다.

연산의 용이성을 위해 모든 변수의 평균은 0이라는 가정을 합니다.

                                                                    $\sum_{r=1}^{n}x_{ri} = 0,\ i=1,2,...,p$

                                           $d_{rs}^2 = (\mathbf{x}_{r}-\mathbf{x}_{s})^T(\mathbf{x}_{r}-\mathbf{x}_{s})= \mathbf{x}_{r}^T\mathbf{x}_{r}+\mathbf{x}_{s}^T\mathbf{x}_{s}-2\mathbf{x}_{r}^T\mathbf{x}_{s}$

이제 Inner product matrix를 만들기 위해 식을 정리합니다.

먼저 $r$에 대해서 평균:

${1\over n}\sum_{r=1}^{n}d_{rs}^2 = {1 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{r} + {1 \over n}\sum_{r=1}^{n}\mathbf{x}_{s}^T\mathbf{x}_{s} - {2 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{s}$

                              $= {1 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{r} + \mathbf{x}_{s}^T\mathbf{x}_{s}$, since $\sum_{r=1}^{n}\mathbf{x}_{ri} = 0$

위 식을 다시 정리하면,

                                                               $\mathbf{x}_s^T\mathbf{x}_{s} = {1 \over n}\sum_{r=1}^{n}d_{rs}^2-{1 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_r$

$s$에 대해서 평균:

${1\over n}\sum_{s=1}^{n}d_{rs}^2 = {1 \over n}\sum_{s=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{r} + {1 \over n}\sum_{s=1}^{n}\mathbf{x}_{s}^T\mathbf{x}_{s} - {2 \over n}\sum_{s=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{s}$

                              $= \mathbf{x}_{r}^T\mathbf{x}_{r} + {1 \over n}\sum_{s=1}^{n}\mathbf{x}_{s}^T\mathbf{x}_{s}$

마찬가지로 정리하면,

                                                                $\mathbf{x}_r^T\mathbf{x}_{r} = {1 \over n}\sum_{s=1}^{n}d_{rs}^2-{1 \over n}\sum_{s=1}^{n}\mathbf{x}_{s}^T\mathbf{x}_s$

${1 \over n^2}\sum_{r=1}^{n}\sum_{r=1}^{n}d_{rs}^2 = {1 \over n^2}\sum_{r=1}^{n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{r} + {1 \over n^2}\sum_{r=1}^{n}\sum_{r=1}^{n}\mathbf{x}_{s}^T\mathbf{x}_{s} -$

                                                      ${2 \over n^2}\sum_{r=1}^{n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{s}$

                                                $= {1 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{r} + {1 \over n}\sum_{s=1}^{n}\mathbf{x}_{s}^T\mathbf{x}_{s}$

                                                $= {2 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_r$, 첨자만 다를 뿐이므로

                                                                 ${2 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{r} = {1 \over n^2}\sum_{r=1}^{n}\sum_{s=1}^{n}d_{rs}^{2}$

$\mathbf{D}$를 정리하여 구한 $\mathbf{x}_r^T\mathbf{x}_{r}$, $\mathbf{x}_s^T\mathbf{x}_{s}$, ${2 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^T\mathbf{x}_{r}$를 이용하여 $b_{rs}$를 구합니다.

$b_{rs} = \mathbf{x}_{r}^T\mathbf{x}_s$

         $= -{1 \over 2}(d_{rs}^2 - \mathbf{x}_{r}^T\mathbf{x}_{r} - \mathbf{x}_{s}^T\mathbf{x}_{s})$

         $=-{1 \over 2}(d_{rs}^2 - {1 \over n}\sum_{s=1}^{n}d_{rs}^2 + {1\over n}\sum_{s=1}^{n}\mathbf{x}_{s}^T\mathbf{x}_{s} - {1 \over n}\sum_{r=1}^{n}d_{rs}^2 +{1 \over n}\sum_{r=1}^{n}\mathbf{x}_{r}^{T}\mathbf{x}_{r})$

         $=-{1 \over 2}(d_{rs}^2 - {1 \over n}\sum_{s=1}^{n}d_{rs}^2 - {1 \over n}\sum_{r=1}^{n}d_{rs}^2 +{1 \over n^2}\sum_{r=1}^{n}\sum_{s=1}^{n}d_{rs}^2$

         $=a_{rs} -a_{r\cdot}-a_{\cdot s}+a_{\cdot \cdot}$

where $a_{rs}=-{1 \over 2}d_{rs}^2,\ a_{r\cdot} = {1 \over n}\sum_{s}a_{rs},\ a_{\cdot s} = {1 \over n}\sum_{r}a_{rs},\ a_{\cdot \cdot} = {1 \over n^2}\sum_{r}\sum_{s}a_{rs}$

$\mathbf{D}$에서 우리가 알 수 있는 정보는 $d_{rs}^2$밖에 없으므로 $b_{rs}$를 $d_{rs}^2$에 대한 정보로 모두 변환합니다.

결국 마지막 식 $a_{rs} -a_{r\cdot}-a_{\cdot s}+a_{\cdot \cdot}$를 통해 inner product matrix의 각 항목의 값들은 distance matrix의 각 항목의 값들의 선형 결합으로 표현 가능하다는 것을 의미합니다.

최종적으로 $[\mathbf{A}]_{rs} = a_{rs}$을 얻게 되어 $\mathbf{A}$ 와 $\mathbf{H}=\mathbf{I}-{1 \over n}\mathbf{11}^T$라는 행렬을 통해 $\mathbf{B}=\mathbf{HAH}$를 구하게 됩니다. $(\mathbf{D} \Rightarrow \mathbf{B})$

마지막으로 $\mathbf{B}$를 이용하여 $\mathbf{X}$를 구합니다.

$\mathbf{X}$를 $n \times p$ matrix$(p < n)$라고 할 때,

$\mathbf{B} = \mathbf{XX}^T \rightarrow\ rank(\mathbf{B}) = rank(\mathbf{XX}^T) = rank(\mathbf{X})=p$ 가 됩니다.

두 벡터의 내적이므로 $\mathbf{B}$는 symmetric이고 positive semi-definite하며 rank는 $p$ 인 상태입니다. 따라서 Eigen-decomposition에 의해서 $p$개의 non-negative eigenvalues와 $(n - p)$개의 zero eigenvalues를 가지게 됩니다. 따라서,

$\mathbf{B} = \mathbf{V}\Lambda\mathbf{V}^T,\hspace{0.2cm} \Lambda=diag(\lambda_1, \lambda_2, \cdots, \lambda_n), \hspace{0.2cm} \mathbf{V} =[\mathbf{v}_1, \mathbf{v}_2, \cdots, \mathbf{v}_n]$ 로 정리되는데,

$(n-p)$개의 zero eigenvalues를 가지기 때문에,

$\mathbf{B}_1 = \mathbf{V}_1\Lambda_1\mathbf{V}_1^T,\hspace{0.2cm} \Lambda_1=diag(\lambda_1, \lambda_2, \cdots, \lambda_p), \hspace{0.2cm} \mathbf{V}_1 =[\mathbf{v}_1, \mathbf{v}_2, \cdots, \mathbf{v}_p]$로 정리가 됩니다.

따라서 우리가 최종적으로 원하는 $\mathbf{X}$는

                                                                        $\mathbf{X}\mathbf{X}^T = (\mathbf{V}_1\Lambda_1^{1\over 2})(\mathbf{V}_1\Lambda_1^{1\over 2})^T$가 되므로

                                                                                        $\therefore \mathbf{X} = \mathbf{V}_1\Lambda_1^{1 \over 2}$ $(\mathbf{B} \Rightarrow \mathbf{X})$

다시 정리하자면, MDS는 거리(유사도) 정보 행렬 $\mathbf{D}$를 이용하여 $d_{rs}^2$을 구하고 내적 행렬 $\mathbf{B}$를 $d_{rs}^2$의 선형 결합으로 나타내어 저차원의 선형 주축으로 이루어진 좌표계 $\mathbf{X}$를 도출하는 것입니다.
