📄 Manual Evaluation Report: Adaptive RAG vs Baseline RAG

⸻

1. Overview

This evaluation compares Baseline RAG and Adaptive RAG (with compression) across three research papers using manually verified answers.

Each paper consists of 10 queries, evaluated for:

* Answer correctness
* Token usage
* Latency
* Compression behavior

⸻

📊 2. Paper-wise Raw Data

⸻

🧪 Paper 1

Accuracy Evaluation

Metric	Value
Total Queries	10
Baseline Correct	9
Adaptive Correct	9
Baseline Accuracy	90%
Adaptive Accuracy	90%
Improvements	0
Same	10
Degradations	0

⸻

Token Reduction (%)

35.3, 34.4, 24.0, 29.8, 24.3, 26.4, 37.3, 39.4, 16.8, 36.7

* Average: 30.44%
* Min: 16.8%
* Max: 39.4%

⸻

Latency Difference (seconds)

-0.69, +0.56, +1.00, +0.43, -0.25, -1.77, +1.19, +0.06, +1.19, +0.47

* Average: +0.22s

⸻

Observations

* Adaptive preserves accuracy
* Significant token reduction (~30%)
* Slight latency increase

⸻

🧪 Paper 2

Accuracy Evaluation

Metric	Value
Total Queries	10
Baseline Correct	5
Adaptive Correct	9
Baseline Accuracy	50%
Adaptive Accuracy	90%
Improvements	4
Same	5
Degradations	1

⸻

Token Reduction (%)

33.8, 26.7, 33.5, 39.1, 34.4, 26.8, 16.4, 30.1, 35.6, 31.6

* Average: 30.8%

⸻

Latency Difference (seconds)

-4.34, -1.09, -0.50, +0.01, +0.42, -1.81, -1.00, -0.32, -0.85, -0.84

* Average: ~ -1.0s (faster overall)

⸻

Observations

* Major improvement in weak retrieval cases
* Adaptive fixes “Not Found” issues
* Faster and more efficient

⸻

🧪 Paper 3

Accuracy Evaluation

Metric	Value
Total Queries	10
Baseline Correct	9
Adaptive Correct	8
Baseline Accuracy	90%
Adaptive Accuracy	80%
Improvements	0
Same	8
Degradations	2

⸻

Token Reduction (%)

~32–35% (consistent across queries)

* Average: ~33–35%

⸻

Latency Difference

* Slight improvements in some queries
* Mixed overall behavior

⸻

Observations

* Minor drop in accuracy
* Caused by over-compression (loss of details)
* Still strong token efficiency

⸻

📊 3. Cross-Paper Summary

⸻

Accuracy Comparison

Model	Avg Accuracy
Baseline RAG	76.7%
Adaptive RAG	86.7%

👉 Net Gain: +10%

⸻

Token Reduction

Paper	Reduction
Paper 1	30.4%
Paper 2	30.8%
Paper 3	~34%

👉 Overall Average: ~32%

⸻

Latency

Paper	Trend
Paper 1	Slight increase
Paper 2	Faster
Paper 3	Mixed

⸻

🧠 4. Key Insights

✅ Strengths of Adaptive RAG

* Improves performance in weak retrieval scenarios
* Reduces token usage significantly (~30–35%)
* Maintains accuracy in most cases

⚠️ Limitations

* Can lose fine details due to compression
* Slight latency overhead in some cases

⸻

🧾 5. Final Conclusion

Adaptive RAG demonstrates a strong balance between efficiency and accuracy. Across 30 evaluated queries, it improves average accuracy from 76.7% to 86.7% while reducing token usage by approximately 32%. The approach is particularly effective in retrieval-limited scenarios, though minor accuracy degradation may occur due to aggressive compression.
