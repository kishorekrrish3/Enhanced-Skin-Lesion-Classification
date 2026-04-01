try:
    import core.run_pipeline
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
