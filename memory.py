import os

JOURNAL_FILE = os.path.expanduser("~/fire/master_journal.txt")

def read_journal():
    if not os.path.exists(JOURNAL_FILE):
        return "Just started. Need to explore and defeat the next gym."
    with open(JOURNAL_FILE, "r") as f:
        return f.read().strip()

def write_journal(content):
    with open(JOURNAL_FILE, "w") as f:
        f.write(content)

class Memory:
    def __init__(self):
        self.scratchpad = "Started up. No immediate memory."

    def get_journal(self):
        return read_journal()

    def update_journal(self, text):
        write_journal(text)

    def get_scratchpad(self):
        return self.scratchpad

    def update_scratchpad(self, text):
        self.scratchpad = text
