import tkinter as tk
from gui import FEM_GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = FEM_GUI(root)
    root.mainloop()