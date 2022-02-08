using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using System.Linq;

namespace EmotionFerPlus
{

    sealed class Test : MonoBehaviour
    {
        #region Editable attributes

        [SerializeField] NNModel _model = null;
        [SerializeField] ComputeShader _preprocessor = null;
        [SerializeField] UnityEngine.UI.RawImage _preview = null;
        [SerializeField] UnityEngine.UI.Text _label = null;

        #endregion

        #region Compile-time constants

        const int ImageSize = 64;

        WebCamTexture _webCamTexture = null;

        private IWorker worker = null;

        readonly static string[] Labels =
          { "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt"};

        private int _cameraIndex = 0;

        #endregion

        #region MonoBehaviour implementation

        void Start()
        {
            // Create Worker
            worker = ModelLoader.Load(_model).CreateWorker();

            // start capture from WebCam
            WebCamDevice[] devices = WebCamTexture.devices;

            _webCamTexture = new WebCamTexture(
                devices[_cameraIndex].name,
                640, 640, 30);
            _preview.texture = _webCamTexture;
            _webCamTexture.Play();
        }

        private void Update()
        {
            // Change webcamera
            if (Input.GetKeyDown("space"))
            {
                WebCamDevice[] devices = WebCamTexture.devices;
                _cameraIndex ++;
                if(_cameraIndex >= devices.Length){
                    _cameraIndex = 0;
                }
                if(_webCamTexture != null){
                    _webCamTexture.Stop();
                }
                _webCamTexture = new WebCamTexture(
                    devices[_cameraIndex].name,
                    640, 640, 30);
                _preview.texture = _webCamTexture;
                _webCamTexture.Play();
            }

            // Preprocessing
            using var preprocessed = new ComputeBuffer(ImageSize * ImageSize, sizeof(float));
            _preprocessor.SetTexture(0, "_Texture", _preview.texture);
            _preprocessor.SetBuffer(0, "_Tensor", preprocessed);
            _preprocessor.Dispatch(0, ImageSize / 8, ImageSize / 8, 1);

            // Emotion recognition model
            using (var tensor = new Tensor(1, ImageSize, ImageSize, 1, preprocessed))
                worker.Execute(tensor);

            // Output aggregation
            var probs = worker.PeekOutput().AsFloats().Select(x => Mathf.Exp(x));
            var sum = probs.Sum();
            var lines = Labels.Zip(probs, (l, p) => $"{l,-12}: {p / sum:0.00}");
            _label.text = string.Join("\n", lines);
        }

        #endregion
    }

} // namespace EmotionFerPlus
