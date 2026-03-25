import sys
import json
import os
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QSplitter, QLabel, QLineEdit,
    QPushButton, QScrollArea, QFrame, QTextEdit,
    QDialog, QDialogButtonBox, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.json")
USER_KEY = "local_user"

# --- Jreve backend ---
try:
    sys.path.append(os.path.dirname(__file__))
    from jreve_v026 import (
        process_message, shutdown_and_save,
        self_modify, get_profile, save_profiles, load_profiles,
        is_identity_contaminated,
        CONF_THRESHOLD
    )
    KNIGHT_AVAILABLE = True
except ImportError:
    KNIGHT_AVAILABLE = False

# --- Worker ---
class KnightWorker(QThread):
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, text, in_memory_history=None, awaiting_clarification=False, session_id=None):
        super().__init__()
        self.text = text
        self.in_memory_history = in_memory_history or []
        self.awaiting_clarification = awaiting_clarification
        self.session_id = session_id

    def run(self):
        try:
            result = process_message(
                self.text,
                USER_KEY,
                in_memory_history=self.in_memory_history,
                awaiting_clarification=self.awaiting_clarification,
                session_id=self.session_id
            )
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")

class ShutdownWorker(QThread):
    """Silent background save when Jreve closes, then exit"""
    done = pyqtSignal()

    def __init__(self, session_id=None):
        super().__init__()
        self.session_id = session_id

    def run(self):
        try:
            shutdown_and_save(USER_KEY, session_id=self.session_id)
        except Exception:
            pass
        self.done.emit()

# --- Tasks JSON helpers ---
def load_tasks_json():
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_tasks_json(tasks):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

def delete_task_from_json(goal_name):
    tasks = load_tasks_json()
    if USER_KEY in tasks:
        tasks[USER_KEY] = [t for t in tasks[USER_KEY] if t.get("goal") != goal_name]
        save_tasks_json(tasks)

def edit_task_in_json(old_name, new_name):
    tasks = load_tasks_json()
    if USER_KEY in tasks:
        for t in tasks[USER_KEY]:
            if t.get("goal") == old_name:
                t["goal"] = new_name
        save_tasks_json(tasks)

# --- Dialogs ---
class EditTaskDialog(QDialog):
    def __init__(self, current_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Task")
        self.setFixedSize(320, 120)
        self.setStyleSheet("background: #f5f5f5; color: #111111;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.input = QLineEdit(current_name)
        self.input.setStyleSheet("""
            QLineEdit {
                background: #fcfcfc; color: #111111;
                border: 1px solid #444; border-radius: 6px;
                padding: 6px 10px; font-size: 13px;
            }
        """)
        layout.addWidget(self.input)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.setStyleSheet("""
            QPushButton {
                background: #333; color: #ffffff;
                border: 1px solid #444; border-radius: 4px;
                padding: 4px 16px; font-size: 12px;
            }
            QPushButton:hover { background: #444; }
        """)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_value(self):
        return self.input.text().strip()

class OnboardingDialog(QDialog):
    """Asks for the user's name during first launch"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Jreve")
        self.setFixedSize(360, 140)
        self.setStyleSheet("background: #f5f5f5; color: #111111;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        label = QLabel("How would you like Knight to address you?")
        label.setStyleSheet("color: #222; font-size: 13px;")
        layout.addWidget(label)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter your name...")
        self.input.setStyleSheet("""
            QLineEdit {
                background: #e8e8e8; color: #111;
                border: 1px solid #ccc; border-radius: 6px;
                padding: 6px 10px; font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #333; }
        """)
        layout.addWidget(self.input)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.setStyleSheet("""
            QPushButton {
                background: #111; color: #fff;
                border: none; border-radius: 4px;
                padding: 6px 20px; font-size: 12px;
            }
            QPushButton:hover { background: #333; }
        """)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def get_name(self):
        return self.input.text().strip()

# --- Task Row ---
class TaskRow(QWidget):
    deleted = pyqtSignal(str)
    edited = pyqtSignal(str)

    def __init__(self, goal, tension=0.0, status="!", parent=None):
        super().__init__(parent)
        self.goal = goal
        self.name_label = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(6)

        if status in ("Urgent", "紧急"):
            dot_color = "#e24b4a"
        elif status in ("!", "Needs Attention", "需要关注"):
            dot_color = "#ef9f27"
        else:
            dot_color = "#1d9e75"

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {dot_color}; font-size: 8px; background: transparent;")
        dot.setFixedWidth(12)
        layout.addWidget(dot)

        self.name_label = QLabel(goal)
        self.name_label.setStyleSheet("color: #222; font-size: 12px; background: transparent;")
        layout.addWidget(self.name_label, 1)

        if tension is not None:
            val = QLabel(f"{tension:.2f}" if isinstance(tension, float) else "")
            val.setStyleSheet(f"color: {dot_color}; font-size: 11px; background: transparent;")
            val.setFixedWidth(30)
            layout.addWidget(val)

        edit_btn = QPushButton("✎")
        edit_btn.setFixedSize(18, 18)
        edit_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #666; border: none; }
            QPushButton:hover { color: #222; }
        """)
        edit_btn.clicked.connect(self.handle_edit)
        layout.addWidget(edit_btn)

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(18, 18)
        del_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #666; border: none; }
            QPushButton:hover { color: #e24b4a; }
        """)
        del_btn.clicked.connect(self.handle_delete)
        layout.addWidget(del_btn)

        self.setStyleSheet("QWidget { background: transparent; } QWidget:hover { background: #eee; }")
        self.setFixedHeight(34)

    def handle_delete(self):
        reply = QMessageBox.question(
            self, "Delete Task", f"Are you sure you want to delete '{self.goal}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            delete_task_from_json(self.goal)
            self.deleted.emit(self.goal)
            self.setVisible(False)
            self.deleteLater()

    def handle_edit(self):
        dialog = EditTaskDialog(self.goal, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name = dialog.get_value()
            if new_name and new_name != self.goal:
                edit_task_in_json(self.goal, new_name)
                self.goal = new_name
                if self.name_label:
                    self.name_label.setText(new_name)
                self.edited.emit(new_name)

# --- Info Row ---
class InfoRow(QWidget):
    added = pyqtSignal(str)
    ignored = pyqtSignal()

    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.message = message
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(6)

        msg = QLabel(message)
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #222; font-size: 11px; background: transparent;")
        layout.addWidget(msg)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        add_btn = QPushButton("Add to Tasks")
        add_btn.setFixedHeight(22)
        add_btn.setStyleSheet("""
            QPushButton { background: #1d9e75; color: white; border-radius: 4px; padding: 0 8px; font-size: 10px; }
            QPushButton:hover { background: #17856a; }
        """)
        add_btn.clicked.connect(lambda: self.added.emit(self.message))
        btn_row.addWidget(add_btn)

        ign_btn = QPushButton("Ignore")
        ign_btn.setFixedHeight(22)
        ign_btn.setStyleSheet("""
            QPushButton { background: #e0e0e0; color: #555; border-radius: 4px; padding: 0 8px; font-size: 10px; }
            QPushButton:hover { background: #ccc; }
        """)
        ign_btn.clicked.connect(self.ignored.emit)
        btn_row.addWidget(ign_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)
        self.setStyleSheet("QWidget { background: transparent; }")

# --- Collapsible Section ---
class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.expanded = True
        self.item_count = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QWidget()
        self.header.setFixedHeight(32)
        self.header.setStyleSheet("QWidget { background: #f5f5f5; } QWidget:hover { background: #ececec; }")
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)

        h_layout = QHBoxLayout(self.header)
        h_layout.setContentsMargins(14, 0, 10, 0)
        h_layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #222; font-size: 10px; font-weight: 600; letter-spacing: 1px;")
        h_layout.addWidget(self.title_label)

        h_layout.addStretch()

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #666; font-size: 10px; background: transparent;")
        h_layout.addWidget(self.count_label)

        self.arrow = QLabel("▼")
        self.arrow.setStyleSheet("color: #666; font-size: 10px; background: transparent;")
        self.arrow.setFixedWidth(12)
        h_layout.addWidget(self.arrow)

        self.header.mousePressEvent = self.toggle
        outer.addWidget(self.header)

        self.content = QWidget()
        self.content.setStyleSheet("background: transparent;")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        outer.addWidget(self.content)

    def toggle(self, event=None):
        self.expanded = not self.expanded
        self.content.setVisible(self.expanded)
        self.arrow.setText("▼" if self.expanded else "▶")

    def update_count(self, n):
        self.item_count = n
        self.count_label.setText(f"({n})" if n > 0 else "")

    def add_widget(self, widget):
        if self.item_count > 0:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFixedHeight(1)
            sep.setStyleSheet("background: #e8e8e8; border: none;")
            self.content_layout.addWidget(sep)
        
        self.content_layout.addWidget(widget)
        self.item_count += 1
        self.update_count(self.item_count)

    def remove_widget(self, widget):
        """Called when TaskRow is deleted to maintain item count"""
        widget.setVisible(False)
        widget.deleteLater()
        self.item_count = max(0, self.item_count - 1)
        self.update_count(self.item_count)

# --- Chat Bubble ---
class ChatBubble(QWidget):
    def __init__(self, sender, message, meta_tag=None, parent=None):
        """meta_tag: Optional label shown below the bubble (e.g., 'Daily · sonnet 0.62')"""
        super().__init__(parent)
        is_user = (sender == "You")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 4, 16, 2)
        outer.setSpacing(2)

        bubble_row = QHBoxLayout()
        bubble_row.setSpacing(0)

        bubble = QLabel(message)
        bubble.setWordWrap(True)
        bubble.setTextFormat(Qt.TextFormat.RichText)
        bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        bubble.setMaximumWidth(560)

        if is_user:
            bubble.setStyleSheet("""
                QLabel {
                    background: #111111;
                    color: #ffffff;
                    border-radius: 14px;
                    padding: 10px 14px;
                    font-size: 13px;
                    line-height: 1.6;
                }
            """)
            bubble_row.addStretch()
            bubble_row.addWidget(bubble)
        else:
            bubble.setStyleSheet("""
                QLabel {
                    background: #f0f0f0;
                    color: #111111;
                    border-radius: 14px;
                    padding: 10px 14px;
                    font-size: 13px;
                    line-height: 1.6;
                }
            """)
            bubble_row.addWidget(bubble)
            bubble_row.addStretch()

        outer.addLayout(bubble_row)

        # meta tag (only shown for Knight replies)
        if meta_tag and not is_user:
            tag_row = QHBoxLayout()
            tag_label = QLabel(meta_tag)
            tag_label.setStyleSheet("color: #aaa; font-size: 9px; background: transparent; padding-left: 4px;")
            tag_row.addWidget(tag_label)
            tag_row.addStretch()
            outer.addLayout(tag_row)

# --- Main Window ---
class JrevApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jreve")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        
        # --- Conversation History (Memory layer, last 3 rounds) ---
        self.conversation_history: list[dict] = []
        self.awaiting_clarification = False
        self.session_id = f"session_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}_{USER_KEY}"
        
        # --- Worker references (prevent GC) ---
        self.worker = None
        self.shutdown_worker = None
        self._is_closing = False
        
        self.setup_ui()
        self.apply_theme()
        self._run_onboarding()
        self._load_existing_tasks()

    # --- Theme ---
    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #ffffff; color: #111111; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #f0f0f0; width: 4px; border-radius: 2px; }
            QScrollBar::handle:vertical { background: #ccc; border-radius: 2px; }
        """)

    # --- Onboarding ---
    def _run_onboarding(self):
        """Asks for name on first launch and writes to user profile"""
        if not KNIGHT_AVAILABLE:
            return
        profile = get_profile(USER_KEY)
        if profile["identity"]["name"] is None:
            dialog = OnboardingDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                name = dialog.get_name()
                if name:
                    profiles = load_profiles()
                    profiles[USER_KEY]["identity"]["name"] = name
                    save_profiles(profiles)
                    self.append_bubble("Jreve", f"Nice to meet you, {name}. How can I help you?")

    # --- Load existing tasks from tasks.json ---
    def _load_existing_tasks(self):
        tasks = load_tasks_json()
        user_tasks = tasks.get(USER_KEY, [])
        for t in user_tasks:
            self.add_task(t.get("goal", ""), 0.0, " ")

    # --- UI Setup ---
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(48)
        title_bar.setStyleSheet("background: #f5f5f5; border-bottom: 1px solid #e0e0e0;")
        tl = QHBoxLayout(title_bar)
        tl.setContentsMargins(20, 0, 20, 0)

        title_label = QLabel("Jreve")
        title_label.setFont(QFont("SF Pro Display", 16, QFont.Weight.Medium))
        title_label.setStyleSheet("color: #111111; letter-spacing: 2px;")
        tl.addWidget(title_label)
        tl.addStretch()

        self.status_dot = QLabel("● active")
        self.status_dot.setStyleSheet("color: #1d9e75; font-size: 12px;")
        tl.addWidget(self.status_dot)
        main_layout.addWidget(title_bar)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #e0e0e0; }")

        # --- Left panel ---
        left_panel = QWidget()
        left_panel.setFixedWidth(240)
        left_panel.setStyleSheet("background: #f5f5f5; border-right: 1px solid #e0e0e0;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        left_content = QWidget()
        left_content.setStyleSheet("background: #f5f5f5;")
        lc_layout = QVBoxLayout(left_content)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.setSpacing(0)

        self.task_section = CollapsibleSection("TASKS")
        lc_layout.addWidget(self.task_section)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet("background: #e0e0e0; border: none;")
        lc_layout.addWidget(div)

        self.info_section = CollapsibleSection("NEW INFO")
        lc_layout.addWidget(self.info_section)
        
        lc_layout.addStretch()
        left_scroll.setWidget(left_content)
        left_layout.addWidget(left_scroll)
        splitter.addWidget(left_panel)

        # --- Right panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setStyleSheet("background: #ffffff; border: none;")

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background: #ffffff;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(0, 16, 0, 16)
        self.chat_layout.setSpacing(4)
        self.chat_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_container)
        right_layout.addWidget(self.chat_scroll, 1)

        # Input bar
        input_bar = QWidget()
        input_bar.setFixedHeight(60)
        input_bar.setStyleSheet("background: #f5f5f5; border-top: 1px solid #e0e0e0;")
        il = QHBoxLayout(input_bar)
        il.setContentsMargins(16, 10, 16, 10)
        il.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Tell Jreve something...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: #fcfcfc; color: #111111;
                border: 1px solid #ccc; border-radius: 6px;
                padding: 0 12px; font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #333; }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        il.addWidget(self.input_field, 1)

        send_btn = QPushButton("Send")
        send_btn.setFixedSize(64, 36)
        send_btn.setStyleSheet("""
            QPushButton {
                background: #111; color: #fff;
                border-radius: 6px; font-size: 13px;
                font-weight: 500; border: none;
            }
            QPushButton:hover { background: #333; }
        """)
        send_btn.clicked.connect(self.send_message)
        il.addWidget(send_btn)

        right_layout.addWidget(input_bar)
        splitter.addWidget(right_panel)
        splitter.setSizes([240, 860])
        main_layout.addWidget(splitter, 1)

    # --- Task helpers ---
    def add_task(self, goal, tension=0.0, status=" "):
        row = TaskRow(goal, tension, status)
        row.deleted.connect(self.on_task_deleted)
        row.edited.connect(self.on_task_edited)
        self.task_section.add_widget(row)

    def add_new_info(self, message):
        row = InfoRow(message)
        row.added.connect(self.on_info_added)
        row.ignored.connect(lambda: self.on_info_ignored(row))
        self.info_section.add_widget(row)

    def on_task_deleted(self, goal):
        self.append_bubble("Jreve", f"Deleted task: '{goal}'.")

    def on_task_edited(self, new_name):
        self.append_bubble("Jreve", f"Task updated to: '{new_name}'.")

    def on_info_added(self, message):
        self.add_task(message, 0.5, " ")
        self.info_section.item_count = max(0, self.info_section.item_count - 1)
        self.info_section.update_count(self.info_section.item_count)
        self.append_bubble("Jreve", "Added to task list, tracking started.")

    def on_info_ignored(self, row):
        self.info_section.remove_widget(row)

    # --- Chat helpers ---
    def append_bubble(self, sender, message, meta_tag=None):
        bubble = ChatBubble(sender, message, meta_tag=meta_tag)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        QApplication.processEvents()
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )

    def _remove_thinking_bubble(self):
        for i in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(i)
            if item and item.widget():
                label = item.widget().findChild(QLabel)
                if label and "Analyzing" in label.text():
                    item.widget().deleteLater()
                    break

    def _md_to_html(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = text.replace('\n', '<br>')
        return text

    # --- Send ---
    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.status_dot.setText("● thinking")
        self.status_dot.setStyleSheet("color: #ef9f27; font-size: 12px;")
        
        self.append_bubble("You", text)
        self.append_bubble("Jreve", "Analyzing...")

        self.worker = KnightWorker(
            text,
            in_memory_history=self.conversation_history[-3:],
            awaiting_clarification=self.awaiting_clarification,
            session_id=self.session_id
        )
        self.worker.result_ready.connect(self.on_knight_reply)
        self.worker.error.connect(self.on_knight_error)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

    def _on_worker_done(self):
        self.input_field.setEnabled(True)
        self.status_dot.setText("● active")
        self.status_dot.setStyleSheet("color: #1d9e75; font-size: 12px;")

    # --- Knight reply ---
    def on_knight_reply(self, result: dict):
        self._remove_thinking_bubble()
        response = result.get("response", "")
        intent = result.get("intent", "")
        folder = result.get("folder", "")
        model_used = result.get("model_used", "")
        tension = result.get("tension")
        ask_clarification = result.get("ask_clarification", False)
        task_added = result.get("task_added", False)
        goal = result.get("goal")
        status = result.get("status", "")

        # --- self_modify interception ---
        if response == "__SELF_MODIFY__":
            reply = QMessageBox.warning(
                self, "Code Modification Confirmation",
                "Knight is requesting to modify its own code.\n\nProceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.append_bubble("Jreve", "Modifying code...")
                try:
                    msg = self_modify(self.input_field.text() or "")
                    self.append_bubble("Jreve", msg)
                except Exception as e:
                    self.append_bubble("Jreve", f"Modification failed: {e}")
            else:
                self.append_bubble("Jreve", "Code modification cancelled.")
            return

        # --- Update clarification state ---
        self.awaiting_clarification = ask_clarification

        # --- Auto-sync task list ---
        if task_added and goal:
            self.add_task(goal, tension or 0.0, status or " ")

        # --- Meta tag ---
        parts = []
        if folder: parts.append(f"{folder}")
        if model_used: parts.append(model_used)
        if tension is not None: parts.append(f"V={tension:.2f}")
        meta_tag = " · ".join(parts) if parts else None

        # --- Show bubble ---
        html_response = self._md_to_html(response)
        self.append_bubble("Jreve", html_response, meta_tag=meta_tag)

        # --- Update history (filter identity contamination) ---
        if not (KNIGHT_AVAILABLE and is_identity_contaminated(response)):
            self.conversation_history.append({"user": self.input_field.text(), "assistant": response})
        else:
            self.conversation_history.append({"user": self.input_field.text(), "assistant": "(Filtered)"})
            
        if len(self.conversation_history) > 3:
            self.conversation_history = self.conversation_history[-3:]

    def on_knight_error(self, error_msg):
        self._remove_thinking_bubble()
        self.append_bubble("Jreve", f"Error: {error_msg}")

    # --- Close event ---
    def closeEvent(self, event):
        if self._is_closing:
            event.accept()
            return
            
        self._is_closing = True
        event.ignore() # Wait for Knight to finish saving
        self.shutdown_worker = ShutdownWorker(session_id=self.session_id)
        self.shutdown_worker.done.connect(self._on_shutdown_done)
        self.shutdown_worker.start()

    def _on_shutdown_done(self):
        self._is_closing = False
        QApplication.quit()

# Entry
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = JrevApp()
    window.show()
    sys.exit(app.exec())
