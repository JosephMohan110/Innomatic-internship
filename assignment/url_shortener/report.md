# URL Shortener Project Report

## 1. Introduction
The objective of this project was to develop a URL Shortener web application. This application allows users to shorten long URLs into manageable links and keeps a history of all shortened URLs. The project consists of a Flask backend and a modern HTML/CSS frontend.

## 2. Approach & Architecture
I adopted a Model-View-Controller (MVC) like architecture using Flask:
- **Backend (Controller & Model)**: Flask handles the routing and logic. SQLAlchemy is used as the ORM to interact with the SQLite database.
- **Frontend (View)**: HTML templates rendered with Jinja2, styled with Bootstrap 5 and custom CSS for a premium "glassmorphism" look.

### Key Components:
- **Database Model (`URL`)**: Stores `id`, `original_url`, `short_id`, and `created_at`.
- **Shortening Logic**: A random string generator creates a unique 6-character ID for each URL.
- **Routes**:
    - `/`: Handles both displaying the form and processing the shortening request.
    - `/<short_id>`: Looks up the original URL and redirects the user.
    - `/history`: Fetches all records from the database and displays them.

## 3. Implementation Details

### Database
I used `Flask-SQLAlchemy` for database management. The `URL` model ensures that each short ID is unique.

### Backend Logic
- `generate_short_id()`: Generates a random string of 6 alphanumeric characters. It checks against the database to ensure uniqueness.
- **Error Handling**: Uses `first_or_404()` to handle cases where a short ID does not exist.

### Frontend Design
- **Bootstrap 5**: Used for responsive layout and pre-built components like the navbar and tables.
- **Custom CSS**: Implemented a dark theme with glassmorphism effects (translucent backgrounds with blur) to meet the "premium design" requirement.
- **JavaScript**: Added a simple function `copyToClipboard()` to improve user experience.

## 4. Challenges & Solutions
- **Unique IDs**: Ensuring that generated IDs are unique.
    - *Solution*: A `while` loop checks the database for the existence of the ID before assigning it.
- **Duplicate URLs**: Deciding whether to create a new short link for an already existing URL.
    - *Decision*: For simplicity and tracking individual creates, I allowed creating multiple short links for the same original URL.

## 5. Conclusion
The project successfully meets all objectives. It provides a simple, efficient, and visually appealing way to shorten URLs and track them.

## 6. Future Enhancements
- User Authentication (Login/Signup).
- Analytics (Click tracking).
- Custom alias selection.
