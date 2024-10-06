import streamlit as st
import sqlite3
import face_recognition
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import io

class AttendanceSystem:
    def __init__(self):
        self.conn = self.connect_db()
        if self.conn:
            self.known_face_encodings, self.known_face_names = self.load_known_faces()
        else:
            self.known_face_encodings, self.known_face_names = [], []

    def connect_db(self):
        try:
            conn = sqlite3.connect('attendance.db', check_same_thread=False)
            c = conn.cursor()
            # Create staff table
            c.execute('''CREATE TABLE IF NOT EXISTS staff (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            age INTEGER,
                            position TEXT,
                            image_path TEXT,
                            face_encoding BLOB
                        )''')
            # Create attendance table
            c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            staff_id INTEGER,
                            date_time TEXT NOT NULL,
                            status TEXT
                        )''')
            conn.commit()
            st.success("Database connection established and tables ensured")
            return conn
        except sqlite3.Error as e:
            st.error(f"Error creating tables: {e}")
            return None

    def add_staff(self, name, age, position, image_file):
        if self.conn is not None:
            cursor = self.conn.cursor()
            try:
                image_bytes = image_file.read()
                face_image = face_recognition.load_image_file(io.BytesIO(image_bytes))
                face_encodings = face_recognition.face_encodings(face_image)

                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    cursor.execute('''INSERT INTO staff (name, age, position, image_path, face_encoding)
                                      VALUES (?, ?, ?, ?, ?)''',
                                   (name, age, position, image_file.name, face_encoding.tobytes()))
                    self.conn.commit()
                    st.success(f"Staff {name} added successfully")
                    # Reload known faces after adding a new staff
                    self.known_face_encodings, self.known_face_names = self.load_known_faces()
                else:
                    st.error(f"No face detected in the image for {name}. Please upload a clear image.")
            except Exception as e:
                st.error(f"Error adding staff {name}: {e}")
        else:
            st.error("Database connection not established.")

    def load_known_faces(self):
        if self.conn is not None:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name, face_encoding FROM staff")
            staff_data = cursor.fetchall()

            known_face_encodings = []
            known_face_names = []

            for name, face_encoding_bytes in staff_data:
                if face_encoding_bytes:
                    face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float64)
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                else:
                    st.warning(f"Warning: No face encoding found for {name}")

            return known_face_encodings, known_face_names
        else:
            st.error("Error connecting to the database.")
            return [], []

    def record_attendance(self, name):
        if self.conn is not None:
            cursor = self.conn.cursor()
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            try:
                cursor.execute("SELECT id FROM staff WHERE name = ?", (name,))
                staff_id = cursor.fetchone()
                if staff_id:
                    cursor.execute("INSERT INTO attendance (staff_id, date_time, status) VALUES (?, ?, ?)",
                                   (staff_id[0], date_time, "Present"))
                    self.conn.commit()
                    st.success(f"Attendance recorded for {name}")
                else:
                    st.error(f"Staff member {name} not found in the database.")
            except sqlite3.Error as e:
                st.error(f"Error recording attendance: {e}")

    def get_staff_info(self, name):
        if self.conn is not None:
            cursor = self.conn.cursor()
            try:
                cursor.execute("SELECT name, age, position, image_path FROM staff WHERE name = ?", (name,))
                staff_info = cursor.fetchone()
                if staff_info:
                    return staff_info
                else:
                    st.warning(f"No staff member found with name {name}")
                    return None
            except sqlite3.Error as e:
                st.error(f"Error fetching staff info: {e}")
        return None

    def generate_report(self):
        if self.conn is not None:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT staff.name, attendance.date_time, attendance.status 
                              FROM attendance 
                              JOIN staff ON staff.id = attendance.staff_id''')
            attendance_records = cursor.fetchall()
            return attendance_records
        else:
            st.error("Error connecting to the database.")
            return []

def main():
    st.set_page_config(page_title="JP Face Recognize Attendance System", layout="wide")
    st.title("JP Face Recognize Attendance System")

    attendance_system = AttendanceSystem()

    menu = ["Add Staff", "Take Attendance", "Generate Report"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Add Staff":
        st.subheader("Add New Staff")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=18, max_value=100)
        position = st.text_input("Position")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

        if st.button("Add Staff"):
            if name and age and position and image_file:
                attendance_system.add_staff(name, int(age), position, image_file)
            else:
                st.warning("Please fill all fields and upload an image.")

    elif choice == "Take Attendance":
        st.subheader("Attendance System")
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(attendance_system.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(attendance_system.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = attendance_system.known_face_names[best_match_index]

                # Draw a box around the face and label it
                top, right, bottom, left = face_location
                cv2.rectangle(cv2_img, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(cv2_img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(cv2_img, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                if name != "Unknown":
                    attendance_system.record_attendance(name)
                    staff_info = attendance_system.get_staff_info(name)
                    if staff_info:
                        st.write(f"Name: {staff_info[0]}")
                        st.write(f"Age: {staff_info[1]}")
                        st.write(f"Position: {staff_info[2]}")
                        st.write("Status: Present")

            st.image(cv2_img, channels="BGR")

    elif choice == "Generate Report":
        st.subheader("Attendance Report")
        attendance_records = attendance_system.generate_report()
        if attendance_records:
            st.table(attendance_records)
        else:
            st.info("No attendance records found.")

if __name__ == "__main__":
    main()
