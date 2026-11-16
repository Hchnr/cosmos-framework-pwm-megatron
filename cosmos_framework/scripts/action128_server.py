"""Start the Action128 WAM HTTP server; see docs/action_serving.md."""

if __name__ == "__main__":
    import os

    os.environ["COSMOS_TRAINING"] = "0"
    from cosmos_framework.inference.action_server import main

    main()
