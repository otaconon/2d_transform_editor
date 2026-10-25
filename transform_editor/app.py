from transform_editor import engine


def main():
  with engine.Window("App") as win:
    if win == None: return
    while win.render():
      pass
  

if __name__ == "__main__":
  main()