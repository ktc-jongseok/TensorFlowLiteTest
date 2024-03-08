package com.example.tensorflowlitetest

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import coil.compose.rememberImagePainter
import coil.request.ImageRequest
import com.example.tensorflowlitetest.ui.theme.TensorFlowLiteTestTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {
    private var tflite: Interpreter? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            TensorFlowLiteTestTheme {
                // A surface container using the 'background' color from the theme
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    tflite = Interpreter(loadModelFile(applicationContext.assets, "ml_data.tflite"))
                    tflite?.let {
                        ImageComparisonScreen(it)
                    }
                }
            }
        }
    }
}

fun resizeImage(bitmap: Bitmap, reqWidth: Int, reqHeight: Int): Bitmap? {
    val aspectRatio: Float = bitmap.width.toFloat() / bitmap.height.toFloat()
    var finalWidth = reqWidth
    var finalHeight = reqHeight

    // 비율을 유지하면서 크기 조정
    if (bitmap.width > bitmap.height) {
        // 가로가 세로보다 길 경우
        finalHeight = (finalWidth / aspectRatio).toInt()
    } else {
        // 세로가 가로보다 길 경우
        finalWidth = (finalHeight * aspectRatio).toInt()
    }

    // 이미지 리사이징
    return Bitmap.createScaledBitmap(bitmap, finalWidth, finalHeight, true)
}

fun preprocessImage(bitmap: Bitmap): TensorImage {
    // TensorImage 초기화 및 로드
    val tensorImage = TensorImage(DataType.FLOAT32)
    tensorImage.load(bitmap)

    // 이미지 전처리를 위한 ImageProcessor 구성
    val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)) // 예: 224x224로 리사이즈
        .build()

    // 전처리된 이미지 반환
    return imageProcessor.process(tensorImage)
}

// 모델 파일 로드
fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
    val fileDescriptor = assetManager.openFd(modelPath)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

// 특징 추출
fun extractFeatures(tensorImage: TensorImage, tflite: Interpreter): FloatArray {
    // 출력을 위한 버퍼 준비
    val output = Array(1) { FloatArray(1001) } // MobileNetV3의 출력 크기에 따라 조정
    // 이미지 예측
    tflite.run(tensorImage.buffer, output)

    return output[0]
}

private fun calculateCosineSimilarity(vectorA: FloatArray, vectorB: FloatArray): Float {
    // 코사인 유사도 계산
    var dotProduct = 0f
    var normA = 0f
    var normB = 0f
    for (i in vectorA.indices) {
        dotProduct += vectorA[i] * vectorB[i]
        normA += vectorA[i] * vectorA[i]
        normB += vectorB[i] * vectorB[i]
    }
    Log.d("Calculate","Cos sim A : $normA ,  sim B : $normB")
    return dotProduct / (sqrt(normA.toDouble()) * sqrt(normB.toDouble())).toFloat()
}

@Composable
fun ImageComparisonScreen(interpreter: Interpreter) {
    val context = LocalContext.current
    var imageUri1 by remember { mutableStateOf<Uri?>(null) }
    var imageUri2 by remember { mutableStateOf<Uri?>(null) }
    var resultText by remember { mutableStateOf<String?>(null) }

    // 이미지 선택 결과 처리를 위한 런처
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        // imageUri1 또는 imageUri2에 URI 할당하는 로직 필요
        if (imageUri1 == null) {
            imageUri1 = uri
        } else if (imageUri2 == null) {
            imageUri2 = uri
            // 이미지 두 장이 모두 선택되면 유사도 계산 함수 호출
            imageUri1?.let { uri1 ->
                imageUri2?.let { uri2 ->
                    val featureVector1 = extractFeatures(preprocessImage(uri1.toBitmap(context)), interpreter)
                    val featureVector2 = extractFeatures(preprocessImage(uri2.toBitmap(context)), interpreter)

                    calculateCosineSimilarity(featureVector1, featureVector2).toPercentage().let {
                        resultText = it.toString()
                    }
                }
            }
        }
    }

    LazyColumn(modifier = Modifier.padding(16.dp)) {
        item {
            Button(
                onClick = {
                    launcher.launch("image/*")
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
            ) {
                Text("Select First Image")
            }
        }

        item(imageUri1) {
            imageUri1?.toBitmap(context).also { bitmapImage ->
                bitmapImage?.let {
                    val resizedBitmap = resizeImage(it, 800, 600) // 예시 크기
                    AsyncImage(
                        model = ImageRequest.Builder(LocalContext.current)
                            .data(resizedBitmap)
                            .crossfade(true)
                            .build(),
                        placeholder = painterResource(R.drawable.baseline_downloading_24),
                        contentDescription = "Selected Image1",
                        error = painterResource(id = R.drawable.baseline_error_24),
                    )
                }
            }
        }

        item {
            Button(
                onClick = {
                    launcher.launch("image/*")
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
            ) {
                Text("Select Second Image")

            }
        }

        item(imageUri2) {
            imageUri2?.toBitmap(context).also { bitmapImage ->
                bitmapImage?.let {
                    val resizedBitmap = resizeImage(it, 800, 600) // 예시 크기
                    AsyncImage(
                        model = ImageRequest.Builder(LocalContext.current)
                            .data(resizedBitmap)
                            .crossfade(true)
                            .build(),
                        placeholder = painterResource(R.drawable.baseline_downloading_24),
                        contentDescription = "Selected Image1",
                        error = painterResource(id = R.drawable.baseline_error_24),
                    )
                }
            }
        }

        item {
            Button(
                onClick = {
                    imageUri1 = null
                    imageUri2 = null
                    resultText = null
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
            ) {
                Text("Clear Images")
            }
        }

        item {
            resultText?.let {
                Text(
                    text = "Two Pictures similarity is \n$it %",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

fun Uri.toBitmap(context: Context): Bitmap {
    return if (Build.VERSION.SDK_INT < 28) {
        val bitmap = MediaStore.Images.Media.getBitmap(
            context.contentResolver,
            this
        )
        bitmap
    } else {
        val source = ImageDecoder.createSource(context.contentResolver, this)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            decoder.setTargetSampleSize(1) // shrinking by
            decoder.isMutableRequired = true // this resolve the hardware type of bitmap problem
        }
        bitmap
    }
}

fun Float.toPercentage(): Float {
    return ((this + 1) / 2) * 100
}