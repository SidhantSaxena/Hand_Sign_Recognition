# **About**  
It is a web platform app that aims to predict/translate hand signs of people with speech disabilities into words.  

---

# **Modes**  
The user can switch between **Alphabet Mode** and **Gesture Mode** by pressing the **Shift** button on their keyboard.  

---

# **Data Collection**  
Feature extraction is done using **MediaPipe**, where **3D key points** are extracted from both **left and right hands** along with **shoulders**. These points are **normalized with respect to the wrist position by default**, making them more tolerant to variations in hand positions. This normalization improves the robustness of dynamic gesture recognition.  

The extracted features are preprocessed and stored in `.npz` files for the corresponding hand signs.  

This project is trained with hand sign/gesture data for:
- **Words**: "eat," "hello," "help," "thanks," "no," "please," "what," "yes"  
- **Alphabets**: A-Z  

This setup is easily scalable for additional signs and gestures.  

---

# **Model Development**  
The project consists of **two separate models**:  
1. **Alphabet Recognition Model**  
   - Uses **15 frames per sample**  
   - Trained using an **LSTM model**  
2. **Gesture Recognition Model**  
   - Uses **30 frames per sample**  
   - Trained using an **LSTM model**  

Both models are trained using the extracted data and saved inside the **Model** folder.  



---

# **Evaluation**  
- **Gesture Model Evaluation**: Performed inside the **model development notebook**.  
- **Alphabet Model Evaluation**: Conducted separately in the **evaluation notebook**.

---

# **Sample Output**
![](fogif.gif)
