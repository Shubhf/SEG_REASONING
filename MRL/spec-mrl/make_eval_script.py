import py_compile
src_path, dst_path = "run4mteb_SpecMRL-alldims.py", "run4mteb_custom.py"
s = open(src_path).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, f"pattern not found exactly once: {old[:60]!r}"
    s = s.replace(old, new)
rep('    parser.add_argument("--output_dir", type=str, default="./output/SpecMRL")\n',
    '    parser.add_argument("--output_dir", type=str, default="./output/SpecMRL")\n'
    '    parser.add_argument("--mrl", action="store_true",\n'
    '                        help="plain MRL checkpoint: identity basis (prefix = first m coords)")\n'
    '    parser.add_argument("--subset15", action="store_true",\n'
    '                        help="paper Table 14 15-task subset")\n')
rep('        projection_basis = load_aligned_projection_basis(\n            proj_head_path, last_vh_path\n        )\n',
    '        if args.mrl:\n'
    '            projection_basis = torch.eye(config.hidden_size)\n'
    '        else:\n'
    '            projection_basis = load_aligned_projection_basis(\n'
    '                proj_head_path, last_vh_path\n'
    '            )\n')
rep('    run_task_groups(ALL_TASKS, model, args, model_name)\n',
    '    tasks = ALL_TASKS\n'
    '    if args.subset15:\n'
    '        keep = {"STS": 3, "Classification": 3, "Clustering": 3, "PairClassification": 1,\n'
    '                "Reranking": 2, "Retrieval": 3}\n'
    '        tasks = [(t, l[:keep[t]]) for t, l in ALL_TASKS if keep.get(t, 0) > 0]\n'
    '    run_task_groups(tasks, model, args, model_name)\n')
open(dst_path, "w").write(s)
py_compile.compile(dst_path, doraise=True)
print("ok:", dst_path)
