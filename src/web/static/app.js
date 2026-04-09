(function () {
  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
      return;
    }
    fn();
  }

  onReady(function () {
    const datasetSelect = document.getElementById("dataset");
    const samplePathInput = document.getElementById("sample_path");
    const sampleFileInput = document.getElementById("sample_file");
    const dropZone = document.getElementById("drop_zone");
    const browseButton = document.getElementById("browse_file_btn");
    const selectedFileName = document.getElementById("selected_file_name");
    const datasetHelp = document.getElementById("dataset_help");

    if (!datasetSelect || !samplePathInput || !sampleFileInput || !dropZone || !browseButton || !selectedFileName || !datasetHelp) {
      return;
    }

    const setSelectedFileLabel = function () {
      if (!sampleFileInput.files || sampleFileInput.files.length === 0) {
        selectedFileName.textContent = "No file selected.";
        return;
      }
      selectedFileName.textContent = "Selected: " + sampleFileInput.files[0].name;
    };

    const setDatasetGuidance = function () {
      const dataset = String(datasetSelect.value || "").toLowerCase();
      const uploadEnabled = dataset === "ixi";

      sampleFileInput.disabled = !uploadEnabled;
      browseButton.disabled = !uploadEnabled;
      dropZone.classList.toggle("disabled", !uploadEnabled);

      if (uploadEnabled) {
        datasetHelp.textContent = "IXI: use sample path or drag-and-drop a .nii/.nii.gz file.";
        samplePathInput.placeholder = "/abs/path/to/IXI123-Guys-0000-T2.nii.gz";
      } else {
        datasetHelp.textContent = "BraTS: use sample path to a patient directory containing flair/t1/t1ce/t2 files.";
        samplePathInput.placeholder = "/abs/path/to/sample_dir";
        sampleFileInput.value = "";
      }

      setSelectedFileLabel();
    };

    const preventDefaults = function (event) {
      event.preventDefault();
      event.stopPropagation();
    };

    ["dragenter", "dragover", "dragleave", "drop"].forEach(function (eventName) {
      dropZone.addEventListener(eventName, preventDefaults);
    });

    ["dragenter", "dragover"].forEach(function (eventName) {
      dropZone.addEventListener(eventName, function () {
        if (!sampleFileInput.disabled) {
          dropZone.classList.add("drag-over");
        }
      });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
      dropZone.addEventListener(eventName, function () {
        dropZone.classList.remove("drag-over");
      });
    });

    dropZone.addEventListener("drop", function (event) {
      if (sampleFileInput.disabled) {
        return;
      }

      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || files.length === 0) {
        return;
      }

      const transfer = new DataTransfer();
      transfer.items.add(files[0]);
      sampleFileInput.files = transfer.files;
      setSelectedFileLabel();
    });

    dropZone.addEventListener("click", function (event) {
      const target = event.target;
      if (target && target.id === "browse_file_btn") {
        return;
      }
      if (!sampleFileInput.disabled) {
        sampleFileInput.click();
      }
    });

    dropZone.addEventListener("keypress", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        preventDefaults(event);
        if (!sampleFileInput.disabled) {
          sampleFileInput.click();
        }
      }
    });

    browseButton.addEventListener("click", function () {
      if (!sampleFileInput.disabled) {
        sampleFileInput.click();
      }
    });

    sampleFileInput.addEventListener("change", setSelectedFileLabel);
    datasetSelect.addEventListener("change", setDatasetGuidance);

    setDatasetGuidance();
  });
})();
