#!/usr/bin/env python3
"""Read retained diagnostics and deletion ledgers; distinguish held errors/events."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ERROR=re.compile(rb'"background_error"\s*:\s*(null|"(?:[^"\\]|\\.)*")')
PEAKS=re.compile(rb'"(queue_depth|max_queue_depth|max_pipe_write_ms|state_writer_queue_depth|dropped_frames)"\s*:\s*([0-9.]+)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,default=Path('docs/audit/runtime_log_statistics.json'))
    args=parser.parse_args()
    ROOT=args.root.resolve()
    output=args.out.resolve()
    if any(output.is_relative_to(ROOT/base) for base in ['yam_data_collection/data','yam_single_arm_collection/data']):
        raise SystemExit('Output must be outside the input data directories')
    paths=sorted(p for base in ['yam_data_collection/data','yam_single_arm_collection/data'] for p in (ROOT/base).rglob('*.jsonl'))
    counts=Counter();transitions=Counter();examples={};peaks={};total=0
    for p in paths:
        previous=None
        with p.open('rb') as f:
            for line_no,line in enumerate(f,1):
                total+=1
                m=ERROR.search(line)
                error=json.loads(m[1]) if m else None
                if error:
                    category=re.sub(r'\d+(?:\.\d+)?','<n>',error)
                    counts[category]+=1
                    if error!=previous:transitions[category]+=1
                    if category not in examples:
                        row=json.loads(line);c=row.get('collector',{})
                        examples[category]={'path':str(p.relative_to(ROOT)),'line':line_no,'message':error,'episode_id':c.get('episode_id'),'state':c.get('state'),'snapshot_unix_ns':c.get('snapshot_unix_ns')}
                previous=error
                for k,v in PEAKS.findall(line):
                    key=k.decode();value=float(v)
                    if value>peaks.get(key,{}).get('value',-1):peaks[key]={'value':value,'path':str(p.relative_to(ROOT)),'line':line_no}
    ledgers=[];reasons=Counter();seconds=Counter()
    for base in ['yam_data_collection/data','yam_single_arm_collection/data']:
        for p in sorted((ROOT/base).rglob('deleted_episode_durations.txt')):
            rows=0
            for line in p.read_text().splitlines():
                cols=line.split('\t')
                if len(cols)<4 or cols[2] not in ['validation_failed','recording_error','user_deleted']:continue
                reasons[cols[2]]+=1;seconds[cols[2]]+=float(cols[3]);rows+=1
            ledgers.append({'path':str(p.relative_to(ROOT)),'rows':rows})
    result={'files':len(paths),'lines':total,'error_status_samples':dict(counts),'error_transitions':dict(transitions),'examples':examples,'global_peaks':peaks,'deletion_ledgers':ledgers,'deletion_reasons':dict(reasons),'deletion_duration_s':dict(seconds),
            'limitations':'An error can persist through many idle status samples. Transition counts are diagnostic observations, not proven independent incidents. Ledgers only cover periods after the feature was introduced and retained files.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'files':len(paths),'lines':total,'error_categories':len(counts),'deletion_reasons':dict(reasons),'peaks':peaks},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
