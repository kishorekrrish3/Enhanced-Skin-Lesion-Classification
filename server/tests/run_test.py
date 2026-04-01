import traceback
try:
    import core.run_pipeline
except Exception as e:
    with open('proper_error_log.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
