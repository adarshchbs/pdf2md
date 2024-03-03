

<template>
  <div id="app" class="container mt-5">
    <div class="row justify-content-center">
      <div class="col-md-6">
        <h1 class="text-center mb-4">PDF Upload</h1>
        <div class="card p-4">
          <div class="card-body">
            <input
              type="file"
              @change="onFileSelected"
              :disabled="uploading"
              accept=".pdf"
              class="form-control-file mb-3"
            />
            <button
              v-if="selectedFile"
              @click="uploadPDF"
              :disabled="uploading"
              class="btn btn-primary w-100"
            >
              Upload
            </button>
            <div v-if="uploading" class="text-center my-2">
              Uploading...
            </div>
            <!-- Display the success message -->
            <div v-if="uploadResponse" class="alert alert-success" role="alert">
              {{ uploadResponse }}
            </div>
            <!-- Display the processed text -->
            <div v-if="processedText" class="processed-text">
              <h3 class="text-center">Extracted Text:</h3>
              <pre class="bg-light p-3 border">{{ processedText }}</pre>
            </div>
        </div>
      </div>
    </div>
  </div>
</div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'App',
  
  data() {
    return {
      selectedFile: null,
      uploadResponse: null,
      uploading: false,
      processedText: ''
    };
  },
  methods: {
    onFileSelected(event) {
      this.selectedFile = event.target.files[0];
    },
    async uploadPDF() {
      if (!this.selectedFile) return;

      this.uploading = true;
      const formData = new FormData();
      formData.append('file', this.selectedFile);
      
      try {
    const response = await axios.post('http://127.0.0.1:8000/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    this.uploadResponse = "Successfully uploaded the PDF."; // Or you can display the message from the response.
    this.processedText = response.data.processed_text; // Set the processed text
  } catch (error) {
    // Enhanced error handling
    if (error.response && error.response.data) {
          this.uploadResponse = error.response.data.detail || 'An error occurred while processing the response.';
        } else {
          // Handle errors without a response (e.g., network errors)
          this.uploadResponse = error.message || 'An unexpected error occurred.';
        }
  } finally {
    this.uploading = false;
    this.selectedFile = null;
  }
}
  }
};
</script>


<style>
/* Add any additional styles here */
.processed-text pre {
    white-space: pre-wrap; /* Since 'pre' element maintains formatting, this ensures long text wraps properly */
    word-wrap: break-word; /* Breaks the words to prevent overflow */
}
</style>