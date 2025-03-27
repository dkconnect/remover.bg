document.getElementById("imageInput").addEventListener("change", async function () {
    const file = this.files[0];
    if (!file) return;

    // Display the uploaded image in the "Before" section
    const beforeImage = document.getElementById("beforeImage");
    beforeImage.src = URL.createObjectURL(file);

    // Prepare form data to send to backend
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Send image to backend (replace with your deployed backend URL)
        const response = await fetch("https://your-app.onrender.com/remove-background/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Failed to process image");

        // Display the result in the "After" section
        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);
        const afterImage = document.getElementById("afterImage");
        afterImage.src = imageUrl;
    } catch (error) {
        console.error("Error:", error);
        alert("Failed to remove background. Please try again.");
    }
});
