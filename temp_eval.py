import json
import glob
import statistics

for f in sorted(glob.glob('evaluation_results/paper*_results.json')):
    data = json.load(open(f))
    reductions = [q['metrics']['token_reduction'] for q in data['results']]
    latencies = [q['metrics']['latency_diff'] for q in data['results']]
    print(f"{f}:")
    print(f"  Avg Token Reduction: {statistics.mean(reductions):.2f}%")
    print(f"  Avg Latency Diff: {statistics.mean(latencies):.2f}s")
