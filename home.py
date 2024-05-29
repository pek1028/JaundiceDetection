import datetime
import io
import os
import re
import shutil
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import bcrypt
import cv2
import numpy as np
from PIL import Image, ImageTk
from colorthief import ColorThief
from keras.models import load_model

current_user_id = None
current_user_role = None
model = load_model('model.h5')


def preprocess_eye_for_prediction(eye_roi, target_size=(100, 100)):
    if eye_roi.ndim == 2 or eye_roi.shape[2] == 1:
        eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_GRAY2RGB)
    eye_roi_resized = cv2.resize(eye_roi, target_size)
    eye_roi_normalized = eye_roi_resized / 255.0
    return np.expand_dims(eye_roi_normalized, axis=0)


def start_real_time_detection(username, user_home_window):
    user_home_window.withdraw()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    detection_window = tk.Toplevel()
    detection_window.title("Real-time Detection")
    message_label = tk.Label(detection_window, text="Please stay still in well-lit conditions for accurate prediction.",
                             bg="red")
    message_label.pack(pady=10)

    cap = cv2.VideoCapture(0)
    video_label = tk.Label(detection_window)
    video_label.pack()

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_with_boxes = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame_with_boxes, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = img_tk
            video_label.configure(image=img_tk)
            video_label.after(10, update_frame)
        else:
            messagebox.showerror("Webcam Error", "Failed to capture frame from webcam.")

    def stop_detection():
        cap.release()
        cv2.destroyAllWindows()
        detection_window.destroy()
        user_home_window.deiconify()

    def analyze_face_and_eyes():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        eye_used_for_prediction = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for index, (ex, ey, ew, eh) in enumerate(eyes):
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_prepared = preprocess_eye_for_prediction(eye_roi)
                color_data = get_dominant_color(frame[y + ey:y + ey + eh, x + ex:x + ex + ew])

                if eye_prepared.ndim == 3:
                    eye_prepared = np.expand_dims(eye_prepared, axis=0)
                if color_data.ndim == 1:
                    color_data = np.expand_dims(color_data, axis=0)
                if not eye_used_for_prediction:
                    prediction = model.predict([eye_prepared, color_data])
                    jaundice_detected = prediction[0][0] < 0.3
                    jaundice_score = 100 - prediction[0][0] * 100

                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 3)
                    cv2.putText(frame, "Predicted Eye", (x + ex, y + ey - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    show_captured_frame(frame, username, jaundice_detected, jaundice_score)
                    eye_used_for_prediction = True
                else:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    analyze_face_button = ttk.Button(detection_window, text="Analyze Face and Eyes", command=analyze_face_and_eyes)
    analyze_face_button.pack(pady=10)

    stop_detection_button = ttk.Button(detection_window, text="Stop Detection", command=stop_detection)
    stop_detection_button.pack(pady=10)

    update_frame()


def show_captured_frame(frame, username, jaundice_detected, jaundice_score):
    user_id = get_user_id(username)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=img)

    window = tk.Toplevel()
    window.title("Capture Preview")
    window.configure(bg='#f0f0f0')

    window_width = 800
    window_height = 600
    center_window(window, window_width, window_height)

    label = tk.Label(window, image=photo)
    label.image = photo
    label.pack()

    info_label = tk.Label(window, text=f"Jaundice Detected: {jaundice_detected}, Score: {jaundice_score:.2f}%")
    info_label.pack()

    def save_image():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/user_{user_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        rounded_jaundice_score = round(jaundice_score, 4)
        save_image_details(user_id, filename, rounded_jaundice_score, "Image saved from real-time detection")
        window.destroy()

    save_button = ttk.Button(window, text="Save Image", command=save_image)
    save_button.pack(side=tk.LEFT, padx=10, pady=10)

    discard_button = ttk.Button(window, text="Discard", command=window.destroy)
    discard_button.pack(side=tk.RIGHT, padx=10, pady=10)


def save_image_details(user_id, image_path, jaundice_percentage, other_details):
    try:
        with sqlite3.connect('jaundice_detection.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO images (userid, image_path, jaundice_percentage, timestamp, other_details)
                VALUES (?, ?, ?, ?, ?)""",
                           (user_id, image_path, jaundice_percentage, datetime.datetime.now(), other_details))
            conn.commit()
        messagebox.showinfo("Success", "Image and details saved successfully!")
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to save image and details: {e}")


def get_dominant_color(image_path):
    if isinstance(image_path, str):
        color_thief = ColorThief(image_path)
    elif isinstance(image_path, np.ndarray):
        is_success, buffer = cv2.imencode(".jpg", image_path)
        if not is_success:
            raise ValueError("Could not encode image to buffer")
        byte_stream = io.BytesIO(buffer)
        color_thief = ColorThief(byte_stream)
    else:
        raise ValueError("Input must be a file path or a numpy array")

    dominant_color = color_thief.get_color(quality=1)
    return np.array([float(c) / 255.0 for c in dominant_color])


def update_image_details(image_id, other_details):
    try:
        with sqlite3.connect('jaundice_detection.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE images
                SET other_details = ?
                WHERE image_id = ?
            """, (other_details, image_id))
            conn.commit()
        messagebox.showinfo("Success", "Details updated successfully!")

    except sqlite3.Error as e:
        messagebox.showerror("Error", f"Failed to update details in the database: {e}")


def show_user_homepage(username):
    user_home_window = tk.Toplevel()
    user_home_window.title("Homepage")
    user_home_window.configure(bg='#f0f0f0')

    window_width = 450
    window_height = 400
    center_window(user_home_window, window_width, window_height)

    configure_styles()

    user_id = get_user_id(username)

    start_icon = get_resized_image('Icon/cam.png')
    upload_icon = get_resized_image('Icon/upload.png')
    logout_icon = get_resized_image('Icon/logout.png')
    images_icon = get_resized_image('Icon/album.png')
    edit_icon = get_resized_image('Icon/user.png')

    welcome_label = tk.Label(user_home_window, text=f"Welcome, {username}!", font=('Helvetica', 18, 'bold'), fg='black')
    welcome_label.pack(pady=(0, 20))

    start_detection_button = ttk.Button(user_home_window, text="Start Real-time Detection", image=start_icon,
                                        compound='left',
                                        command=lambda: start_real_time_detection(username, user_home_window))
    start_detection_button.image = start_icon
    start_detection_button.pack(pady=8, fill=tk.X)

    edit_profile_button = ttk.Button(user_home_window, text="Edit Profile", image=edit_icon, compound='left',
                                     command=lambda: edit_profile(username))
    edit_profile_button.image = edit_icon
    edit_profile_button.pack(pady=8, fill=tk.X)

    upload_image_button = ttk.Button(user_home_window, text="Upload Image", image=upload_icon, compound='left',
                                     command=lambda: upload_image(user_id))
    upload_image_button.image = upload_icon
    upload_image_button.pack(pady=8, fill=tk.X)

    show_images_button = ttk.Button(user_home_window, text="Show My Images", image=images_icon, compound='left',
                                    command=lambda: display_images_for_user(user_home_window, user_id))
    show_images_button.image = images_icon
    show_images_button.pack(pady=8, fill=tk.X)

    def user_logout():
        user_home_window.destroy()
        cv2.destroyAllWindows()
        show_login_screen()

    logout_button = ttk.Button(user_home_window, text="Logout", image=logout_icon,
                               compound=tk.LEFT, command=user_logout)
    logout_button.image = logout_icon
    logout_button.pack(pady=8, fill=tk.X)


def upload_image(user_id):
    messagebox.showinfo("Upload Image", "Please upload your eyes to do prediction!")
    filename = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not filename:
        return

    try:
        destination = os.path.join('images', os.path.basename(filename))
        shutil.copy(filename, destination)
        messagebox.showinfo("Image Upload", "Image uploaded successfully!")

        image = cv2.imread(destination)
        eye_roi_resized = cv2.resize(image, (100, 100))
        eye_roi_normalized = eye_roi_resized / 255.0
        eye_roi_batch = np.expand_dims(eye_roi_normalized, axis=0)

        color_thief = ColorThief(destination)
        dominant_color = color_thief.get_color(quality=1)
        color_data = np.array([dominant_color]) / 255.0

        jaundice_probability = model.predict([eye_roi_batch, color_data])
        jaundice_percentage = jaundice_probability[0][0] * 100
        jaundice_percentage = 100 - jaundice_percentage

        formatted_jaundice_score = round(jaundice_percentage, 4)

        save_upload_image(user_id, destination, formatted_jaundice_score, None)
        messagebox.showinfo("Jaundice Prediction", f"Probability of Jaundice: {jaundice_percentage:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to upload and predict: {str(e)}")


def save_upload_image(user_id, image_path, jaundice_percentage, other_details):
    if user_id is None:
        messagebox.showerror("Error", "User ID is None. Cannot save image without a valid user ID.")
        return

    timestamp = datetime.datetime.now()
    try:
        with sqlite3.connect('jaundice_detection.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO images (userid, image_path, jaundice_percentage, timestamp, other_details)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, image_path, jaundice_percentage, timestamp, other_details))
            conn.commit()
            print("Image path and details saved to the database successfully.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        messagebox.showerror("Error", f"Failed to save image to the database: {e}")


def display_images_for_admin(window):
    global root, tree
    root = tk.Toplevel(window)
    root.title("All Images")
    root.configure(bg='#f0f0f0')

    window_width = 950
    window_height = 600
    center_window(root, window_width, window_height)

    configure_styles()

    columns = ('image_id', 'user_id', 'image_path', 'jaundice_percentage', 'timestamp', 'other_details')
    tree = ttk.Treeview(root, columns=columns, show='headings', selectmode='browse')
    for col in columns:
        tree.heading(col, text=col.replace('_', ' ').title())
        tree.column(col, width=120)
    tree.pack(expand=True, fill='both', padx=10, pady=10)

    refresh_data()

    filter_var = tk.StringVar(root)
    filter_options = ["All", "Jaundice Detected", "No Jaundice Detected"]
    filter_dropdown = ttk.Combobox(root, textvariable=filter_var, values=filter_options)
    filter_dropdown.pack(pady=10)

    def apply_filter(event):
        selected_filter = filter_var.get()
        for item in tree.get_children():
            tree.delete(item)
        if selected_filter == "All":
            rows = fetch_image_data()
        elif selected_filter == "Jaundice Detected":
            rows = fetch_image_data(jaundice=True)
        elif selected_filter == "No Jaundice Detected":
            rows = fetch_image_data(jaundice=False)

        for row in rows:
            tree.insert('', tk.END, values=row)

    filter_dropdown.bind('<<ComboboxSelected>>', apply_filter)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill='x')

    view_btn = ttk.Button(btn_frame, text="View Image",
                          command=lambda: view_selected_image(tree))
    view_btn.pack(side='left', padx=10, pady=10)

    add_btn = ttk.Button(btn_frame, text="Add/Edit Details",
                         command=lambda: add_details(tree.set(tree.selection()[0], 'image_id')))
    add_btn.pack(side='left', padx=10, pady=10)

    delete_btn = ttk.Button(btn_frame, text="Delete Image",
                            command=lambda: delete_image_data(tree.set(tree.selection()[0], 'image_id')))
    delete_btn.pack(side='left', padx=10, pady=10)

    close_button = ttk.Button(btn_frame, text="Close", command=root.destroy)
    close_button.pack(side='right', padx=10, pady=10)


def display_images_for_user(window, user_id):
    user_images_window = tk.Toplevel(window)
    user_images_window.title("My Images")
    user_images_window.configure(bg='#f0f0f0')

    window_width = 950
    window_height = 600
    center_window(user_images_window, window_width, window_height)

    configure_styles()

    columns = ('image_id', 'image_path', 'jaundice_percentage', 'timestamp', 'other_details')
    tree = ttk.Treeview(user_images_window, columns=columns, show='headings', selectmode='browse')
    for col in columns:
        tree.heading(col, text=col.replace('_', ' ').capitalize())
        tree.column(col, width=120)

    user_images = fetch_user_image_data(user_id)
    if not user_images:
        print("DEBUG: No images to display.")

    for image in user_images:
        tree.insert('', 'end', values=(image[0], image[1], image[2], image[3], image[4]))

    tree.pack(expand=True, fill='both', padx=20, pady=15)

    view_button = ttk.Button(user_images_window, text="View Selected Image", command=lambda: view_selected_image(tree))
    view_button.pack(pady=10)

    close_button = ttk.Button(user_images_window, text="Close", command=user_images_window.destroy)
    close_button.pack(pady=10)


def view_selected_image(tree):
    selected_item = tree.selection()
    if selected_item:
        item = tree.item(selected_item)
        admin_view = 'user_id' in tree['columns']

        if admin_view:
            image_path = item['values'][2]
            jaundice_percentage = item['values'][3]
            other_details = item['values'][5]
        else:
            image_path = item['values'][1]
            jaundice_percentage = item['values'][2]
            other_details = item['values'][4]

        try:
            view_image(image_path, jaundice_percentage, other_details)
        except Exception as e:
            messagebox.showerror("View Image", f"Could not open the image: {e}")


def refresh_data():
    for item in tree.get_children():
        tree.delete(item)
    for row in fetch_image_data():
        tree.insert('', tk.END, values=row)


def view_image(image_path, jaundice_percentage, other_details):
    window = tk.Toplevel()
    window.title("Image Details")
    window.geometry("800x600")
    window.configure(bg='#f0f0f0')

    window_width = 800
    window_height = 600
    center_window(window, window_width, window_height)

    try:
        img = Image.open(image_path)
        img = img.resize((780, 450), Image.ANTIALIAS)  # Resize image for better display
        photo = ImageTk.PhotoImage(img)

        label_image = tk.Label(window, image=photo)
        label_image.image = photo  # Keep a reference to avoid garbage collection
        label_image.pack(padx=10, pady=10)

        label_details = tk.Label(window,
                                 text=f"Jaundice Percentage: {jaundice_percentage}%, Other Details: {other_details}",
                                 wraplength=700, justify="left")
        label_details.pack(padx=10)

        back_button = ttk.Button(window, text="Close", command=window.destroy)
        back_button.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {str(e)}")


def add_details(image_id):
    def submit_details():
        details = details_entry.get()
        if details:
            update_image_details(image_id, details)
            refresh_data()
            add_window.destroy()
        else:
            messagebox.showerror("Error", "Please enter additional details.")

    add_window = tk.Toplevel()
    add_window.title("Add Image Details")
    add_window.resizable(False, False)
    add_window.configure(bg='#f0f0f0')

    window_width = 300
    window_height = 120
    center_window(add_window, window_width, window_height)

    tk.Label(add_window, text="Enter additional details:").pack(padx=10, pady=(10, 0))

    details_entry = tk.Entry(add_window)
    details_entry.pack(padx=10, pady=5)

    button_frame = ttk.Frame(add_window)
    button_frame.pack(padx=10, pady=10)

    ok_button = ttk.Button(button_frame, text="OK", command=submit_details)
    ok_button.pack(side=tk.LEFT, expand=True, padx=(0, 5))

    cancel_button = ttk.Button(button_frame, text="Cancel", command=add_window.destroy)
    cancel_button.pack(side=tk.RIGHT, expand=True, padx=(5, 0))

    details_entry.focus_set()


def fetch_user_image_data(user_id):
    with sqlite3.connect('jaundice_detection.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
               SELECT image_id, image_path, jaundice_percentage, timestamp, other_details 
               FROM images WHERE userid = ?
           """, (user_id,))
        results = cursor.fetchall()
        print(f"Fetching data for user ID {user_id}: {results}")  # Log the results
        return results


def delete_image_data(image_id):
    if messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete this image record?"):
        with sqlite3.connect('jaundice_detection.db') as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
            conn.commit()
        messagebox.showinfo("Success", "Image record deleted successfully.")
        refresh_data()


def show_admin_page():
    admin_window = tk.Toplevel()
    admin_window.title("Admin Dashboard")
    admin_window.configure(bg='#f0f0f0')

    window_width = 400
    window_height = 340
    center_window(admin_window, window_width, window_height)

    configure_styles()

    logout_icon = get_resized_image('Icon/logout.png')
    images_icon = get_resized_image('Icon/album.png')
    users_icon = get_resized_image('Icon/user.png')

    title_label = tk.Label(admin_window, text="Admin Homepage", font=('Helvetica', 18, 'bold'), fg='black')
    title_label.pack(pady=8, fill=tk.X)

    welcome_label = tk.Label(admin_window, text="The photo of the users should be confidential!",
                             font=('Helvetica', 10), fg='red')
    welcome_label.pack(pady=(0, 20))

    show_images_button = ttk.Button(admin_window, text="Show All Images", image=images_icon,
                                    compound=tk.LEFT, command=lambda: display_images_for_admin(admin_window))
    show_images_button.image = images_icon
    show_images_button.pack(pady=8, fill=tk.X)

    view_users_button = ttk.Button(admin_window, text="View Users", image=users_icon,
                                   compound=tk.LEFT, command=lambda: view_users(admin_window))
    view_users_button.image = users_icon
    view_users_button.pack(pady=8, fill=tk.X)

    def admin_logout():
        admin_window.destroy()
        show_login_screen()

    logout_button = ttk.Button(admin_window, text="Logout", image=logout_icon,
                               compound=tk.LEFT, command=admin_logout)
    logout_button.image = logout_icon
    logout_button.pack(pady=8, fill=tk.X)


def fetch_image_data(jaundice=None):
    conn = sqlite3.connect('jaundice_detection.db')
    cursor = conn.cursor()
    if jaundice is None:
        query = "SELECT image_id, userid, image_path, jaundice_percentage, timestamp, other_details FROM images"
    elif jaundice:
        query = "SELECT image_id, userid, image_path, jaundice_percentage, timestamp, other_details FROM images WHERE jaundice_percentage >= 30"
    else:
        query = "SELECT image_id, userid, image_path, jaundice_percentage, timestamp, other_details FROM images WHERE jaundice_percentage < 30"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_user_id(username):
    with sqlite3.connect('jaundice_detection.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        return result[0] if result else None


def view_users(admin_window):
    users_window = tk.Toplevel(admin_window)
    users_window.title("View Users")
    users_window.configure(bg='#f0f0f0')

    window_width = 400
    window_height = 400
    center_window(users_window, window_width, window_height)

    configure_styles()

    listbox_frame = ttk.Frame(users_window)
    listbox_frame.pack(fill='both', expand=True, padx=10, pady=10)

    users_listbox = tk.Listbox(listbox_frame, height=15, exportselection=0)
    users_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=users_listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    users_listbox['yscrollcommand'] = scrollbar.set

    user_ids = {}

    def refresh_users_listbox():
        users_listbox.delete(0, tk.END)
        user_ids.clear()
        with sqlite3.connect('jaundice_detection.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, username, phone FROM users")
            for user_id, username, phone in cursor.fetchall():
                display_text = f"{username} - {phone}" if phone else f"{username} - No phone"
                users_listbox.insert(tk.END, display_text)
                user_ids[username] = user_id

    refresh_users_listbox()

    def delete_selected_user():
        selection = users_listbox.curselection()
        if not selection:
            messagebox.showinfo("Error", "Please select a user to delete.")
            return
        selected_text = users_listbox.get(selection[0])
        username = selected_text.split(' - ')[0]

        if username in user_ids:
            user_id = user_ids[username]
            delete_user(user_id, username, refresh_users_listbox)
        else:
            messagebox.showerror("Error", f"User {username} not found in the list.")

    def edit_selected_user():
        selection = users_listbox.curselection()
        if not selection:
            messagebox.showinfo("Error", "Please select a user to edit.")
            return
        selected_text = users_listbox.get(selection[0])
        username = selected_text.split(' - ')[0]
        edit_user(username, refresh_users_listbox)

    def go_back():
        users_window.destroy()
        admin_window.deiconify()

    button_frame = ttk.Frame(users_window)
    button_frame.pack(fill='x', pady=10)

    edit_button = ttk.Button(button_frame, text="Edit", command=edit_selected_user)
    edit_button.pack(side=tk.LEFT, padx=10, expand=True)

    delete_button = ttk.Button(button_frame, text="Delete", command=delete_selected_user)
    delete_button.pack(side=tk.LEFT, padx=10, expand=True)

    back_button = ttk.Button(button_frame, text="Back", command=go_back)
    back_button.pack(side=tk.RIGHT, padx=10, expand=True)

    users_window.protocol("WM_DELETE_WINDOW", users_window.destroy)


def configure_styles():
    style = ttk.Style()
    style.theme_use('clam')  # Use the 'clam' theme as a base for custom styling
    style.configure('TButton', font=('Arial', 12), padding=6)
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0', font=('Arial', 12))
    style.configure('TEntry', font=('Arial', 12), padding=6)
    style.configure('TListbox', font=('Arial', 12))


def delete_user(user_id, username, refresh_callback):
    response = messagebox.askyesno("Delete User", f"Are you sure you want to delete {username} and all related data?")
    if response:
        try:
            with sqlite3.connect('jaundice_detection.db') as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE user_id=?", (user_id,))
                conn.commit()
            messagebox.showinfo("Success", f"User {username} and all related data deleted successfully.")
            refresh_callback()
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Failed to delete user {username}: {e}")


def edit_user(username, refresh_callback):
    def submit_new_username():
        new_username = new_username_entry.get()
        if new_username and new_username != username:
            try:
                with sqlite3.connect('jaundice_detection.db') as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE users SET username=? WHERE username=?", (new_username, username))
                    conn.commit()
                messagebox.showinfo("Success", "Username updated successfully.")
                edit_window.destroy()  # Close the dialog
                refresh_callback()  # Refresh the listbox
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Failed to update username. The username might already exist.")
        elif new_username == username:
            messagebox.showinfo("Info", "The username is unchanged.")

    edit_window = tk.Toplevel()
    edit_window.title("Edit User")
    edit_window.resizable(False, False)
    edit_window.configure(bg='#f0f0f0')

    window_width = 300
    window_height = 120
    center_window(edit_window, window_width, window_height)

    tk.Label(edit_window, text="Enter new username:").pack(padx=10, pady=(10, 0))

    new_username_entry = tk.Entry(edit_window)
    new_username_entry.pack(padx=10, pady=5)
    new_username_entry.insert(0, username)

    button_frame = ttk.Frame(edit_window)
    button_frame.pack(padx=10, pady=10)

    ok_button = ttk.Button(button_frame, text="OK", command=submit_new_username)
    ok_button.pack(side=tk.LEFT, expand=True, padx=(0, 5))

    cancel_button = ttk.Button(button_frame, text="Cancel", command=edit_window.destroy)
    cancel_button.pack(side=tk.RIGHT, expand=True, padx=(5, 0))

    new_username_entry.focus_set()


def fetch_user_info(username):
    with sqlite3.connect('jaundice_detection.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT age, phone, medical_conditions
            FROM users WHERE username = ?""", (username,))
        user_info = cursor.fetchone()
    return user_info


def is_valid_age(age):
    if age.isdigit() and 0 <= int(age) <= 100:
        return '1'
    elif age == "":
        return '1'
    else:
        return '0'


def is_valid_phone(phone):
    result = re.match(r'^1(1\d{0,8}|\d{0,8})$', phone) is not None or phone == ""
    print(f"Validating phone '{phone}': {result}")
    return '1' if result else '0'


def edit_profile(username):
    user_info = fetch_user_info(username)

    if not user_info:
        messagebox.showerror("Error", "No data found for the user.")
        return

    def update_info():
        new_password = password_entry.get()
        age = age_entry.get()
        phone = phone_entry.get()
        conditions = conditions_entry.get("1.0", tk.END).strip()

        success = update_user_info(username, new_password, age, phone, conditions)

        if success:
            messagebox.showinfo("Success", "Profile updated successfully.")
        else:
            messagebox.showerror("Error", "Profile update failed.")

    profile_window = tk.Toplevel()
    profile_window.title(f"Edit Profile - {username}")
    profile_window.configure(bg='#f0f0f0')

    window_width = 600
    window_height = 400
    center_window(profile_window, window_width, window_height)

    configure_styles()

    validate_age_cmd = profile_window.register(is_valid_age)
    validate_phone_cmd = profile_window.register(lambda p: is_valid_phone(p))

    tk.Label(profile_window, text="New Password:").pack()
    password_entry = tk.Entry(profile_window, show="*")
    password_entry.pack()

    tk.Label(profile_window, text="Age:").pack()
    age_entry = tk.Entry(profile_window, validate="key", validatecommand=(validate_age_cmd, '%P'))
    age_entry.pack()
    age_entry.insert(0, user_info[0] if user_info[0] is not None else '')

    tk.Label(profile_window, text="Phone Number (+60):").pack()
    phone_entry = tk.Entry(profile_window, validate="key", validatecommand=(validate_phone_cmd, '%P'))
    phone_entry.pack()
    phone_number = user_info[1][3:] if user_info[1] is not None else ''
    phone_entry.insert(0, phone_number)

    tk.Label(profile_window, text="Medical Conditions:").pack()
    conditions_entry = tk.Text(profile_window, height=4, width=50)
    conditions_entry.pack()
    conditions_entry.insert("1.0", user_info[2] if user_info[2] is not None else '')

    update_info_button = ttk.Button(profile_window, text="Update Info", command=update_info)
    update_info_button.pack(pady=10)

    back_button = ttk.Button(profile_window, text="Back", command=lambda: profile_window.destroy())
    back_button.pack(pady=10)


def hash_password(password):
    password_bytes = password.encode('utf-8')  # Convert the password to bytes
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)  # Generate the hashed password
    return hashed_password


def update_user_info(username, new_password, age, phone, conditions):
    password_hash = hash_password(new_password) if new_password else None
    phone = '+60' + phone if not phone.startswith('+60') else phone

    try:
        with sqlite3.connect('jaundice_detection.db') as conn:
            cursor = conn.cursor()
            if password_hash:
                cursor.execute("""
                        UPDATE users 
                        SET 
                            password_hash = ?,
                            age = ?,
                            phone = ?,
                            medical_conditions = ?
                        WHERE username = ?""",
                               (password_hash, age, phone, conditions, username))
            else:
                cursor.execute("""
                        UPDATE users 
                        SET 
                            age = ?,
                            phone = ?,
                            medical_conditions = ?
                        WHERE username = ?""",
                               (age, phone, conditions, username))
            conn.commit()
        return True
    except sqlite3.Error as e:
        print("Database error:", e)
        return False


def get_resized_image(image_path):
    try:
        original_icon = Image.open(image_path)
        resized_icon = original_icon.resize((32, 32), Image.ANTIALIAS)
        tk_resized_icon = ImageTk.PhotoImage(resized_icon)

        return tk_resized_icon
    except Exception as e:
        print(f"Error resizing icon: {e}")
        return None


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    window.geometry(f"{width}x{height}+{x}+{y}")


def show_login_screen():
    login_screen = tk.Tk()
    login_screen.title("Login")
    login_screen.configure(bg='#f0f0f0')
    icon_path = "Icon/icon.ico"
    login_screen.iconbitmap(icon_path)

    window_width = 600
    window_height = 400
    center_window(login_screen, window_width, window_height)

    configure_styles()
    system_label = tk.Label(login_screen, text="Jaundeyes", font=('Helvetica', 20, 'bold'), fg='black')
    system_label.pack(pady=(10, 10))

    title_label = tk.Label(login_screen, text="Login", font=('Helvetica', 18, 'bold'), fg='black')
    title_label.pack(pady=(0, 10))

    login_frame = tk.Frame(login_screen, padx=10, pady=10, bg="#F0F0F0")
    login_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    username_label = tk.Label(login_frame, text="Username:", font=('Helvetica', 12), bg="#F0F0F0")
    username_label.grid(row=0, column=0, sticky="e")
    username_entry = tk.Entry(login_frame, font=('Helvetica', 12), width=20)
    username_entry.grid(row=0, column=1, padx=10, pady=10)

    password_label = tk.Label(login_frame, text="Password:", font=('Helvetica', 12), bg="#F0F0F0")
    password_label.grid(row=1, column=0, sticky="e")
    password_entry = tk.Entry(login_frame, font=('Helvetica', 12), show="*", width=20)
    password_entry.grid(row=1, column=1, padx=10, pady=10)

    def attempt_login():
        ADMIN_USERNAME = "admin"
        ADMIN_PASSWORD = "adminpass"

        entered_username = username_entry.get()
        entered_password = password_entry.get()

        if entered_username == ADMIN_USERNAME and entered_password == ADMIN_PASSWORD:
            messagebox.showinfo("Login Success", "You are now logged in as admin.")
            login_screen.destroy()
            show_admin_page()
            return

        username = login_user(entered_username, entered_password)
        if username:
            messagebox.showinfo("Success", f"Welcome, {username}!")
            login_screen.destroy()
            show_user_homepage(username)
        else:
            messagebox.showerror("Error", "Invalid username or password")

    login_button = ttk.Button(login_frame, text="Login", width=20, command=lambda: attempt_login())
    login_button.grid(row=2, column=0, columnspan=2, pady=10)

    def open_register_screen():
        register_screen = tk.Toplevel(login_screen)
        register_screen.title("Register")
        register_screen.configure(bg='#f0f0f0')

        center_window(register_screen, window_width, window_height)

        configure_styles()

        title_label = tk.Label(register_screen, text="Register", font=('Helvetica', 18, 'bold'), fg='black')
        title_label.pack(pady=(0, 8))

        register_frame = tk.Frame(register_screen, padx=10, pady=10, bg="#F0F0F0")
        register_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(register_frame, text="Enter username:", bg="#F0F0F0").grid(row=0, column=0, sticky="w")
        new_username_entry = tk.Entry(register_frame)
        new_username_entry.grid(row=0, column=1, pady=5)

        tk.Label(register_frame, text="Enter password:", bg="#F0F0F0").grid(row=1, column=0, sticky="w")
        new_password_entry = tk.Entry(register_frame, show="*")
        new_password_entry.grid(row=1, column=1, pady=5)

        def go_back():
            register_screen.destroy()

        def attempt_register():
            if register_user(new_username_entry.get(), new_password_entry.get()):
                messagebox.showinfo("Success", "Registration successful, please log in.")
                register_screen.destroy()
            else:
                messagebox.showerror("Error", "Registration failed (username might already exist).")

        ttk.Button(register_frame, text="Register", command=attempt_register, width=10).grid(row=2, column=0,
                                                                                             columnspan=2, pady=5)

        back_button = ttk.Button(register_frame, text="Back", command=go_back, width=10)
        back_button.grid(row=3, column=0, columnspan=2, pady=5)

    register_button = ttk.Button(login_frame, text="Redirect to Register", width=20,
                                 command=lambda: open_register_screen())
    register_button.grid(row=3, column=0, columnspan=2, pady=10)

    login_screen.mainloop()


def setup_database():
    with sqlite3.connect('jaundice_detection.db') as conn:
        c = conn.cursor()
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users
            (user_id INTEGER PRIMARY KEY AUTOINCREMENT,
             username TEXT UNIQUE,
             password_hash TEXT,
             age INTEGER DEFAULT NULL,
             phone TEXT DEFAULT NULL,
             medical_conditions TEXT DEFAULT NULL)
        ''')
        # Images table
        c.execute('''
            CREATE TABLE IF NOT EXISTS images
            (image_id INTEGER PRIMARY KEY AUTOINCREMENT,
             userid INTEGER NOT NULL,
             image_path TEXT NOT NULL,
             jaundice_percentage REAL NOT NULL,
             timestamp DATETIME NOT NULL,
             other_details TEXT DEFAULT NULL,
             FOREIGN KEY (userid) REFERENCES users(user_id) ON DELETE CASCADE)
        ''')
        print("Database setup complete.")


def register_user(username, password):
    password_hash = hash_password(password)  # Hash the password
    try:
        with sqlite3.connect('jaundice_detection.db') as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO users (username, password_hash) 
                VALUES (?, ?)
            ''', (username, password_hash))
            conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        print("Failed to insert user:", e)
        return False


def login_user(username, password):
    print(f"Logging in with Username: '{username}'")
    with sqlite3.connect('jaundice_detection.db') as conn:
        username = username.strip().lower()
        cursor = conn.cursor()
        cursor.execute("SELECT username, password_hash FROM users WHERE lower(username)=?", (username,))
        user_record = cursor.fetchone()
        if user_record and bcrypt.checkpw(password.encode('utf-8'), user_record[1]):
            print("Login successful")
            return user_record[0]
        else:
            print("Login failed: Invalid username or password")
            return None


def main_app():
    root = tk.Tk()
    root.withdraw()
    show_login_screen()
    root.mainloop()


if __name__ == "__main__":
    setup_database()
    main_app()
