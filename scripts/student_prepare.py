"""Prepare an approved 16FPS teacher clip and its reference image for overfit diagnostics."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--person-id', required=True)
    parser.add_argument('--approved', action='store_true', help='source pair has been reviewed and matched')
    parser.add_argument('--review-note', required=True)
    args = parser.parse_args()
    if not args.approved:
        parser.error('review the source pair first, then explicitly pass --approved')
    from student.data import prepare
    prepare(args.video, args.reference, args.output, args.person_id, args.review_note)


if __name__ == '__main__':
    main()
