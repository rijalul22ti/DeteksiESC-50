let mediaRecorder;
let audioChunks = [];
let selectedFile = null;

document.addEventListener("DOMContentLoaded", function () {
  // --- REKAM AUDIO ---
  const recordBtn = document.getElementById('recordBtn');
  const stopBtn = document.getElementById('stopBtn');

  recordBtn.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // webm untuk browser compatibility
        const formData = new FormData();
        formData.append("audio", audioBlob, "recorded.webm");

        fetch("/predict", {
          method: "POST",
          body: formData
        })
          .then(response => response.json())
          .then(data => {
            console.log("Prediksi:", data.result);
            $("#hasil").text("Hasil prediksi: " + data.result).show();
          })
          .catch(error => {
            console.error("Error:", error);
            alert("Terjadi kesalahan saat memproses audio.");
          });
      };

      mediaRecorder.start();
      stopBtn.disabled = false;
      recordBtn.disabled = true;
    } catch (err) {
      alert("Gagal mengakses mikrofon.");
      console.error(err);
    }
  });

  stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      stopBtn.disabled = true;
      recordBtn.disabled = false;
    }
  });

  // --- PREDIKSI DARI FILE INPUT ---
  $('#audioInput').on('change', function (event) {
    selectedFile = event.target.files[0];

    if (selectedFile) {
      alert("File audio berhasil dipilih: " + selectedFile.name);
    }
  });

  $('#predictBtn').on('click', function (e) {
    e.preventDefault();

    if (!selectedFile) {
      alert("Silakan pilih file audio terlebih dahulu.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", selectedFile);

    fetch("/predict", {
      method: "POST",
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        console.log("Prediksi:", data.result);
        $("#hasil").text("Hasil prediksi: " + data.result).show();
      })
      .catch(error => {
        console.error("Error:", error);
        alert("Terjadi kesalahan saat memproses audio.");
      });
  });
});
