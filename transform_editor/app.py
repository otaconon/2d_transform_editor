from transform_editor import engine
import argparse

def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="transform editor", description="Polygon drawing typof app")
  parser.add_argument("--window_size", nargs=2, type=int, default=[900, 900], metavar=("WIDTH", "HEIGHT"))
  return parser

def main(argv: list[str] | None = None):
  parser = build_parser()
  args = parser.parse_args(argv)
  with engine.Window("App", *args.window_size) as win:
    if win == None: return
    while win.render():
      pass
  

if __name__ == "__main__":
  main()