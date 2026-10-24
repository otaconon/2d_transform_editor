from transform_editor import engine


def main():
  window = engine.Window("App")
  while window.render():
    pass
  

if __name__ == "__main__":
  main()