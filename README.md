# PhonePro - Your Personal Phone Recommendation System

PhonePro is a Streamlit-based web application that helps users find the best phone based on their preferences. With a user-friendly interface, PhonePro allows you to input your desired phone specifications and suggests the most suitable model using a machine learning algorithm. Additionally, you can search for details about specific phone models within the app.


## Features

- **Phone Recommendation:** Input various phone specifications like price, brand, battery capacity, RAM, and more to receive a recommended phone model that best matches your needs.
- **Search Phone Details:** Explore detailed specifications of specific phone models using the search functionality in the sidebar.
- **User-Friendly Interface:** Simple and intuitive design to make it easy for users to interact with the app.
- **Machine Learning:** Uses cosine similarity to recommend phones based on input features.

## Getting Started

Follow the steps below to run PhonePro on your local machine.

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- Streamlit
- Pandas
- Scikit-learn
- Pickle

## How It Works

### Input Your Preferences

Use the sliders and dropdowns on the main page to specify your phone requirements. You can adjust parameters such as:

- **Price:** The budget range for the phone.
- **Brand Name:** Preferred brand of the phone.
- **5G Support:** Whether the phone should support 5G.
- **Processor Brand:** The brand of the phone's processor.
- **Battery Capacity:** The battery capacity in mAh.
- **RAM Capacity:** The desired amount of RAM.
- **Internal Memory:** The internal storage size.
- **Refresh Rate:** The screen refresh rate in Hz.
- **Operating System:** The OS you prefer.
- **Primary Rear Camera:** The megapixels of the rear camera.
- **Fast Charging:** The fast charging capacity in watts.

### Get Recommendations

Based on your inputs, the app will suggest the phone model that best matches your preferences using cosine similarity.

### View Phone Details

You can view additional details about the recommended phone or search for specific phone models using the search functionality in the sidebar.

## Dataset

The model uses a dataset (`processed_dataset.csv`) that includes features like price, brand, processor, RAM, battery capacity, and more. The original unscaled dataset (`dataset.csv`) is used for displaying details.


## Contact

For any questions or feedback, reach out at: vedant.kale22@pccoepune.org

---

**Made by Vedant**
