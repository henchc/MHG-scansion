from tkinter import *
from tkinter import filedialog
import csv
from tkinter import ttk
from scan_text_new import scan_text


class Window:

    def __init__(self, master):
        self.filename = ""
        scanned_text = StringVar()

        self.text_entry = Text(mainframe, width=75, height=30)
        self.text_entry.grid(column=2, row=1, columnspan=2, rowspan=50, sticky=(W, E))

        ttk.Button(mainframe, text="Scan!", command=self.process_txt).grid(
            column=2, row=51)
        ttk.Button(mainframe, text="Quit", command=self.quit_button).grid(
            column=1, row=3, sticky=(N))

        ttk.Button(mainframe, text="Save Output", command=self.file_save).grid(
            column=1, row=2, sticky=(N))

        ttk.Button(mainframe, text="Clear", command=self.clear_textbox).grid(
            column=3, row=51)

        ttk.Button(mainframe, text="Browse files...",
                   command=self.browsetxt).grid(column=1, row=1, sticky=(W, N))

        # for child in mainframe.winfo_children():
        #     child.grid_configure(padx=10, pady=10)

        self.text_entry.focus()  # cursor starts here

    def browsetxt(self):

        Tk().withdraw()
        self.filename = filedialog.askopenfilename()
        if len(self.filename) > 0:
            with open(self.filename, 'r') as f:
                self.raw_text = f.read()

            import re
            from string import punctuation
            self.raw_text = self.raw_text.lower()
            self.raw_text = re.sub("[0-9]+[a-z]*", "", self.raw_text)
            self.raw_text = ''.join([i for i in self.raw_text if i not in list(punctuation)])
            self.text_entry.insert('1.0', self.raw_text)

    def process_txt(self):
        if len(self.text_entry.get('1.0', END)) > 0:
            self.scanned = scan_text(self.text_entry.get('1.0', END))
        self.text_entry.delete('1.0', END)
        self.text_entry.insert('1.0', self.scanned)

    def file_save(self):
        f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
        if f is None:  # return `None` if dialog closed with "cancel".
            return

        f.write(self.scanned)
        f.close()

    def clear_textbox(self):
        self.text_entry.delete('1.0', END)

    def quit_button(self):
        root.quit()

root = Tk()
root.title("Automatische Mittelhochdeutsche Skandierung")
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)  # expand to take extra space
mainframe.rowconfigure(0, weight=1)  # expand to take extra space
window = Window(root)
root.mainloop()
