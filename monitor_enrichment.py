#!/usr/bin/env python3
"""Monitor enrichment progress."""
import re
import time
from pathlib import Path

output_file = Path("/tmp/claude/-Users-jasonxu-workspace-video-understanding/tasks/b8c05f4.output")

def parse_progress(line):
    """Parse tqdm progress line."""
    # Example: Enriching:   9%|▉         | 32/364 [06:37<1:07:08, 12.13s/it]
    match = re.search(r'(\d+)/(\d+)\s+\[([^\<]+)\<([^\,]+),\s+([\d\.]+)s/it\]', line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        elapsed = match.group(3)
        remaining = match.group(4)
        speed = float(match.group(5))
        percentage = (current / total) * 100
        return {
            'current': current,
            'total': total,
            'percentage': percentage,
            'elapsed': elapsed,
            'remaining': remaining,
            'speed': speed
        }
    return None

# Read last progress line
with open(output_file) as f:
    lines = f.readlines()

# Find last Enriching line
for line in reversed(lines):
    if 'Enriching:' in line:
        progress = parse_progress(line)
        if progress:
            print(f"{'='*70}")
            print(f"📊 ENRICHMENT PROGRESS")
            print(f"{'='*70}")
            print(f"Progress:      {progress['current']}/{progress['total']} segments ({progress['percentage']:.1f}%)")
            print(f"Elapsed time:  {progress['elapsed']}")
            print(f"Remaining:     {progress['remaining']}")
            print(f"Speed:         {progress['speed']:.1f}s per segment")
            print(f"{'='*70}")

            # Estimate completion
            import datetime
            remaining_seconds = progress['speed'] * (progress['total'] - progress['current'])
            completion_time = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)
            print(f"Estimated completion: {completion_time.strftime('%I:%M %p')}")
            print(f"{'='*70}\n")
            break
