import py_compile
path = "run4mteb_custom.py"
s = open(path).read()

def rep(old, new):
    global s
    assert s.count(old) == 1, f"pattern not found exactly once: {old[:70]!r}"
    s = s.replace(old, new)

rep('''                evaluation = MTEB(tasks=[task], task_langs=["en", "eng-Latn"])
                evaluation.run(
                    model,
                    eval_splits=["test"],
                    output_folder=output_folder,
                    verbosity=0,
                    overwrite_results=False,
                )
''', '''                evaluation = MTEB(tasks=[task], task_langs=["en", "eng-Latn"])
                for _t in getattr(evaluation, "tasks", []):
                    # mteb==1.4.0 points SICK-R at a Hub dataset that was removed; use the canonical repo
                    if getattr(_t.metadata, "hf_hub_name", None) == "MMathematica/sickr-sts":
                        _t.metadata.hf_hub_name = "mteb/sickr-sts"
                        _t.metadata.revision = None
                try:
                    evaluation.run(
                        model,
                        eval_splits=["test"],
                        output_folder=output_folder,
                        verbosity=0,
                        overwrite_results=False,
                    )
                except Exception as e:
                    print(f"[eval-failed] {task} dim={dim}: {e!r}", flush=True)
                    FAILED_TASKS.append((task, repr(e)))
                    break
''')
rep('def run_task_groups(task_groups, model, args, model_name):\n',
    'FAILED_TASKS = []\n\n\ndef run_task_groups(task_groups, model, args, model_name):\n')
rep('    run_task_groups(tasks, model, args, model_name)\n',
    '    run_task_groups(tasks, model, args, model_name)\n'
    '    print("FAILED TASKS:", FAILED_TASKS if FAILED_TASKS else "none", flush=True)\n')
open(path, "w").write(s)
py_compile.compile(path, doraise=True)
print("ok")
