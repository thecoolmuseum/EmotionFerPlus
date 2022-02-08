using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using System.Linq;
using Mediapipe.BlazePose;

namespace EmotionFerPlus
{

    sealed class Test : MonoBehaviour
    {
        #region Editable attributes

        [SerializeField] NNModel _model = null;
        [SerializeField] ComputeShader _preprocessor = null;
        [SerializeField] UnityEngine.UI.RawImage _preview = null;
        [SerializeField] UnityEngine.UI.RawImage _subPreview = null;
        [SerializeField] UnityEngine.UI.Text _label = null;
        [SerializeField] BlazePoseResource blazePoseResource;
        [SerializeField] BlazePoseModel poseLandmarkModel;

        #endregion

        #region Compile-time constants

        private BlazePoseDetecter detecter = null;

        const int ImageSize = 64;

        WebCamTexture _webCamTexture = null;
        Texture _tempTexture = null;

        private IWorker worker = null;

        readonly static string[] Labels =
          { "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt"};

        private int _cameraIndex = 0;
        private float fps = 0.0f;
        #endregion

        #region MonoBehaviour implementation

        void Start()
        {
            // Create pose detecter
            detecter = new BlazePoseDetecter(blazePoseResource, poseLandmarkModel);

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

            // Pose detection
            detecter.ProcessImage(_webCamTexture, poseLandmarkModel);
            ComputeBuffer result = detecter.outputBuffer;

            ComputeBuffer worldLandmarkResult = detecter.worldLandmarkBuffer;

            int count = detecter.vertexCount;

            var data = new Vector4[count];
            result.GetData(data);
            // Debug.Log("---");
            var centerPos = data[0];
            var eyeDistance = (data[3]-data[6]).magnitude;
            var pos = eyeDistance + "/" + centerPos;
            var facePixelSize =  Mathf.RoundToInt(2.3f*(eyeDistance) * _webCamTexture.width);
            var clipTex = new Texture2D(facePixelSize, facePixelSize);
            try{
                var pixel = _webCamTexture.GetPixels(
                    Mathf.RoundToInt(centerPos.x * _webCamTexture.width - facePixelSize/2),
                    Mathf.RoundToInt(centerPos.y * _webCamTexture.height - (facePixelSize-0.2f)/2),
                    facePixelSize, 
                    facePixelSize);
                clipTex.SetPixels(pixel);
                clipTex.Apply();    
                _subPreview.texture = clipTex;
            }catch{
            }

            // worldLandmarkResult.GetData(data);
            // Debug.Log("---");
            // foreach(var d in data){
            //     Debug.Log(d);
            // }

            // Preprocessing
            using var preprocessed = new ComputeBuffer(ImageSize * ImageSize, sizeof(float));
            _preprocessor.SetTexture(0, "_Texture", clipTex);
            _preprocessor.SetBuffer(0, "_Tensor", preprocessed);
            _preprocessor.Dispatch(0, ImageSize / 8, ImageSize / 8, 1);

            // Emotion recognition model
            using (var tensor = new Tensor(1, ImageSize, ImageSize, 1, preprocessed))
                worker.Execute(tensor);

            // Output aggregation
            var probs = worker.PeekOutput().AsFloats().Select(x => Mathf.Exp(x));
            var sum = probs.Sum();
            var lines = Labels.Zip(probs, (l, p) => $"{l,-12}: {p / sum:0.00}");

            fps = (fps + 1f / Time.deltaTime)/2.0f;

            _label.text = /*fps + "fps\n" + pos + "\n" +*/ string.Join("\n", lines);
        }

        #endregion
    }

} // namespace EmotionFerPlus
