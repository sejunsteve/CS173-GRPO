"""Plot and describe real logs. Refuse to fabricate a curve when logs are missing."""
import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=Path('output/grpo'))
    p.add_argument('--baseline', type=Path, default=Path('output/baseline/metrics.json'))
    p.add_argument('--after', type=Path, default=Path('output/after/metrics.json'))
    p.add_argument('--output', type=Path, default=Path('reports'))
    args = p.parse_args()
    log_path = args.run / 'metrics.jsonl'
    if not log_path.exists():
        raise SystemExit('No real training log exists; run Scholar training first. No curve was created.')
    records = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    points = [r for r in records if 'reward' in r and 'step' in r]
    if len(points) < 2:
        raise SystemExit('Need at least two logged training-reward points for a curve.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    steps = [r['step'] for r in points]
    rewards = [r['reward'] for r in points]
    smoothed = [mean(rewards[max(0, i-4):i+1]) for i in range(len(rewards))]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    ax.plot(steps, rewards, color='#9daac1', lw=1, alpha=.8, label='Sampled training reward')
    ax.plot(steps, smoothed, color='#214c82', lw=2.3, label='Trailing mean (up to 5 steps)')
    ax.set(xlabel='Optimizer step', ylabel='Mean outcome reward', ylim=(-.02, 1.02),
           title='Access-control GRPO: observed training reward')
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    fig.savefig(args.output / 'training_curve.png', dpi=180)
    plt.close(fig)
    window = min(10, max(1, len(rewards)//3))
    early, late = mean(rewards[:window]), mean(rewards[-window:])
    decreases = sum(b < a - 1e-9 for a, b in zip(rewards, rewards[1:]))
    text = [f'The run has {len(points)} logged reward points through optimizer step {steps[-1]}.',
            f'Mean reward in the first {window} points is {early:.4f}; in the final {window} it is {late:.4f}.',
            f'There are {decreases} downward transitions among {len(points)-1} adjacent points. '
            + ('The observed curve is nondecreasing.' if decreases == 0 else 'Improvement, if present, is noisy rather than monotonic.'),
            'Training reward is computed on sampled training completions. It alone does not establish held-out improvement or convergence.']
    if args.baseline.exists() and args.after.exists():
        base = json.loads(args.baseline.read_text(encoding='utf-8'))
        after = json.loads(args.after.read_text(encoding='utf-8'))
        if base['n'] != after['n'] or base['data'] != after['data']:
            raise ValueError('Before/after evaluation sets differ')
        if base['model'] != after['model'] or base['model_revision'] != after['model_revision']:
            raise ValueError('Before/after base model revisions differ')
        text.append(f"Greedy held-out average reward: {base['average_reward']:.4f} before, {after['average_reward']:.4f} after; "
                    f"exact-match accuracy: {base['exact_match']:.2%} before, {after['exact_match']:.2%} after.")
        delta = after['average_reward'] - base['average_reward']
        text.append('The held-out reward increased in this single run.' if delta > 0 else 'This run did not improve held-out average reward.')
        if delta > 0 and after['exact_match'] <= base['exact_match']:
            text.append('Exact-match accuracy did not improve, so do not equate the reward gain with better rule identification.')
        text.append('Report per-rule results and valid-JSON rate as well; one seed does not establish statistical significance.')
    else:
        text.append('A matched before/after held-out evaluation is still missing; task improvement remains unverified.')
    (args.output / 'convergence.txt').write_text('\n\n'.join(text) + '\n', encoding='utf-8')
    print('\n\n'.join(text))


if __name__ == '__main__':
    main()
