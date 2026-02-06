The algorithmic details of the code in this repository is found in the following paper: 

Varsi, A., Drousiotis, E., Spirakis, P.G., Maskell, S. (2026). A Shared Memory Optimal Parallel Redistribution Algorithm for SMC Samplers with Variable Size Samples. In: Zhang, Y., Hladik, M., Moosaei, H. (eds) Learning and Intelligent Optimization. LION 2025. Lecture Notes in Computer Science, vol 15745. Springer, Cham. https://doi.org/10.1007/978-3-032-09192-5_7.

which can be found at the following link: https://openreview.net/forum?id=S3R0uG8Dta

Please cite it using the following bibitem: 

@InProceedings{10.1007/978-3-032-09192-5_7,
author="Varsi, Alessandro
and Drousiotis, Efthyvoulos
and Spirakis, Paul G.
and Maskell, Simon",
editor="Zhang, Yingqian
and Hladik, Milan
and Moosaei, Hossein",
title="A Shared Memory Optimal Parallel Redistribution Algorithm for SMC Samplers with Variable Size Samples",
booktitle="Learning and Intelligent Optimization",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="97--112",
abstract="Sequential Monte Carlo (SMC) samplers are Bayesian inference methods to draw N random samples from challenging posterior distributions. Their simplicity and competitive accuracy make them popular in various applications of Machine Learning (ML), Bayesian Optimization (BO), and Statistics. In many applications, run-time is critical under strict accuracy requirements, making parallel computing essential. However, an efficient parallelization of SMC depends on how effectively its bottleneck, the redistribution step, is parallelized. This is hard due to workload imbalance across the cores, especially when the samples are of variable-size. A parallel redistribution for variable-size samples was recently proposed for Shared Memory (SM) architectures. This method resizes all samples to the size of the biggest sample, {\$}{\$}{\backslash}overline{\{}M{\}}{\$}{\$}M{\textasciimacron}, and constrains the samples to be indivisible, i.e., forces the cores to redistribute whole samples. This leads to inefficient run-time, and a sub-optimal time complexity, {\$}{\$}O({\backslash}overline{\{}M{\}}{\backslash}log {\_}2N){\$}{\$}O(M{\textasciimacron}log2N). This study addresses the challenge of Optimal Parallel Redistribution (OPR) for variable-size samples. We first prove that OPR for indivisible variable-size samples is NP-complete. Then, we present an OPR algorithm for SM that does not resize the samples and allows cores to redistribute either whole samples or fractions of them. We prove theoretically that this approach achieves optimal {\$}{\$}O({\backslash}hat{\{}M{\}} {\backslash}log {\_}2 N){\$}{\$}O(M^log2N)time complexity, where {\$}{\$}{\backslash}hat{\{}M{\}}{\$}{\$}M^is the average size of the redistributed samples. We also show experimentally that the proposed approach is up to {\$}{\$}10{\backslash}times {\$}{\$}10{\texttimes}faster than the reference method on a 32-core SM machine.",
isbn="978-3-032-09192-5"
}
