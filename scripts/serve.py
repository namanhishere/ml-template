#!/usr/bin/env python
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import os

    os.environ["CKPT_PATH"] = args.ckpt_path
    uvicorn.run("src.serving.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
