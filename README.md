# ماشین یادگیری با TensorFlow.js

## معرفی

در این پروژه یک ماشین یادگیری ساده برای دسته بندی (Classification) تصاویر با استفاده از دوربین میباشد.
در این پروژه از مدل MobileNet و کتابخانه TensorFlow.js استفاده شده و یک پروژه ساده وب میباشد.

## ویژگیها

-   کلاس بندی کردن دو مدل کلاس از تصویر
-   تشخیص تصاویر کلاس بندی شده

## شروع

در شروع پروژه، صفحه اصلی رو مشاهده میکنیم، پس از نمایش اعلان بالای صفحه که نشان دهنده‌ی لود شدن کامل MobileNet هست میتوان از ابزار استفاده کرد

!["pic"](https://github.com/fardezh/tfl/blob/main/assets/First-screen.png)

## روش کار

1. پس از لود شدن باید وب کم و یا دوربین دستگاه خود را با دکمه `Enable Webcam` فعال کنید.
2. پس از آن شی و یا تصویر مورد نظر خود را رو به روی دوربین بگیرید.
3. دکمه ی `Gather Class 1 Data` رو به مقدار دلخواه نگه دارید، ترجیحا برای هر کلاس تا عدد 30 به بالا نگه دارید.
4. پس از آن، شی و یا تصور را عوض و دکمه بعدی `Gather Class 2 Data` را فشار دهید.
5. به مقدار کافی که داده جمع شد روی دکمه `Train & Predict!` کلیک کنید و منتظر بمانید.
6. حالا با نشان دادن هر کدام از شی و تصویر جلوی وب کم، ماشین متوجه میشود که تصویر از کلاس 1 هست یا کلاس 2.
7. برای انجام دوباره روی دکمه `Reset` کلیک کنید.

# مستندات پروژه

قسمت اصلی پروژه در فایل `script.js` قرار دارد ولی از قبل در انتهای فایل html کتابخانه `TensorFlow` ایمپورت شده است.

```html
<script
    src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.11.0/dist/tf.min.js"
    type="text/javascript"
></script>
```

##### در `script.js` :

1. تمام دکمه ها وارد شده است.

```JavaScript
const STATUS = document.getElementById("status");
const VIDEO = document.getElementById("webcam");
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");
const RESET_BUTTON = document.getElementById("reset");
const TRAIN_BUTTON = document.getElementById("train");
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];
```

2. دکمه های اضافه کردن کلاس رجیستر شده، در صورت نیاز میتوان در Html دکمه های بیشتری برای افزودن کلاس های بیشتر اضافه کرد.

```javascript
ENABLE_CAM_BUTTON.addEventListener("click", enableCam);
TRAIN_BUTTON.addEventListener("click", trainAndPredict);
RESET_BUTTON.addEventListener("click", reset);

let dataCollectorButtons = document.querySelectorAll(
    "button.dataCollector"
);

for (let i = 0; i < dataCollectorButtons.length; i++) {
    dataCollectorButtons[i].addEventListener(
        "mousedown",
        gatherDataForClass
    );
    dataCollectorButtons[i].addEventListener(
        "mouseup",
        gatherDataForClass
    );
    dataCollectorButtons[i].addEventListener(
        "touchend",
        gatherDataForClass
    );
    CLASS_NAMES.push(
        dataCollectorButtons[i].getAttribute("data-name")
    );
}
```

3. تابعی برای لود کردن مدل MobileNet بلافاصله اجرا میشود. در این تابع مدل مورد نظر از لینک خوانده شده و در کتابخانه رجیستر میشود تا از آن استفاده شود. در داخل تابع مدل با آرگومان صفر فراخوانی میشود تا به اصطلاح گرم شود. این تکنیک برای کارایی و دقت مدل لازم هست.

```javaScript
async function loadMobileNetFeatureModel() {
    const URL =
        "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

    mobilenet = await tf.loadGraphModel(URL, {
        fromTFHub: true,
    });

    STATUS.innerText = "MobileNet v3 loaded successfully!";

    tf.tidy(function () {
        let answer = mobilenet.predict(
            tf.zeros([
                1,
                MOBILE_NET_INPUT_HEIGHT,
                MOBILE_NET_INPUT_WIDTH,
                3,
            ])
        );

        console.log(answer.shape);
    });
}

loadMobileNetFeatureModel();
```

4. در این بخش یک شبکه عصبی خطی با TensorFlow.js تعریف میشود. با متد add به مدل ساخته شده میتوان لایه هایی با واحد مورد نیاز اضافه کرد، که در این بخش از 128 واحد تشکیل شده است. و گزینه inputShape نشان دهنده این است که داده های ورودی باید 1024 ویژگی داشته باشد.

```JavaScript
let model = tf.sequential();
model.add(
    tf.layers.dense({
        inputShape: [1024],
        units: 128,
        activation: "relu",
    })
);
```

5. لایه دوم به مدل اضافه می‌شود که معمولاً برای وظایف دسته‌بندی استفاده می‌شود. تعداد واحدهای این لایه توسط `CLASS_NAMES.length` تعیین می‌شود که `CLASS_NAMES` تعداد کلاس‌هایی را که می‌خواهید پیش‌بینی کنید، نمایان می‌کند. تابع فعال‌سازی استفاده‌شده softmax است که برای مسائل دسته‌بندی چند کلاسه مناسب است.

```JavaScript
model.add(
    tf.layers.dense({
        units: CLASS_NAMES.length,
        activation: "softmax",
    })
);
```

6. با فراخوانی `model.summary();` اطلاعات خلاصه‌ای از معماری مدل نشان می‌دهد. این خلاصه شامل جزئیاتی از هر لایه در مدل شما، تعداد پارامترها (وزن‌ها و بایاسها) و شکل خروجی از هر لایه است. این اطلاعات مفید برای فهم و بررسی مدل قبل از آموزش و بعد از آموزش می‌باشد.

7. این بخش مربوط به مرحله تنظیم (compile) مدل شبکه عصبی با استفاده از TensorFlow.js است. این مرحله به تعیین جزئیات آموزش مدل از جمله نوع بهینه‌ساز (optimizer)، تابع خطا (loss function) و معیارهای متریک (metrics) مربوط است.

```javascript
model.compile({
    // بهینه‌ساز Adam که نرخ یادگیری را تطبیقی تغییر می‌دهد.
    optimizer: "adam",
    // استفاده از تابع خطا مناسب. اگر تعداد کلاس‌ها 2 باشد، باید از binaryCrossentropy استفاده شود.
    // در غیر این صورت اگر کلاس‌ها بیشتر از 2 باشند، از categoricalCrossentropy استفاده می‌شود.
    loss:
        CLASS_NAMES.length === 2
            ? "binaryCrossentropy"
            : "categoricalCrossentropy",
    // چون این یک مسئله دسته‌بندی است، می‌توان دقت را در گزارش‌ها نیز ثبت کرد!
    metrics: ["accuracy"],
});
```

تابع `optimizer` (بهینه‌ساز): در اینجا از بهینه‌ساز "Adam" استفاده شده است. بهینه‌ساز مسئله‌ای است که وظیفه بهینه‌سازی وزن‌ها و پارامترهای مدل را دارد. "Adam" یکی از بهینه‌سازهای معمولاً مورد استفاده در شبکه‌های عصبی عمیق است.

-   تابع `loss` (تابع خطا): نوع تابع خطا بر اساس تعداد کلاس‌ها تعیین می‌شود. اگر تعداد کلاس‌ها دو باشد، از "binaryCrossentropy" استفاده می‌شود که برای مسائل دسته‌بندی دو کلاسه (دودویی) مناسب است. در غیر این صورت اگر تعداد کلاس‌ها بیشتر از دو باشد، از "categoricalCrossentropy" استفاده می‌شود که برای مسائل دسته‌بندی چند کلاسه مناسب است.

-   تابع `metrics` (معیارها): در این مورد، معیار دقت ("accuracy") به عنوان یک معیار برای ارزیابی عملکرد مدل در طول آموزش مشخص شده است. این معیار نشان می‌دهد که مدل به چه اندازه توانایی پیش‌بینی درست کلاس‌های واقعی را دارد.

8. هنگامی که دکمه جمع آوری اطلاعات کلاس فشرده میشود تابع زیر اجرا میشود:

```JavaScript
function dataGatherLoop() {
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
        let imageFeatures = calculateFeaturesOnCurrentFrame();

        trainingDataInputs.push(imageFeatures);
        trainingDataOutputs.push(gatherDataState);

        if (examplesCount[gatherDataState] === undefined) {
            examplesCount[gatherDataState] = 0;
        }

        examplesCount[gatherDataState]++;

        STATUS.innerText = "";
        for (let n = 0; n < CLASS_NAMES.length; n++) {
            STATUS.innerText +=
                CLASS_NAMES[n] +
                " data count: " +
                examplesCount[n] +
                ". ";
        }

        window.requestAnimationFrame(dataGatherLoop);
    }

}
```

در این تابع، هنگامی که وبکم روشن است، با فراخوانی تابع دیگری به نام `calculateFeaturesOnCurrentFrame` فریم فعلی تصویر را پردازش و مقادیر آن را به آرایه از پیش درست شده ای اضافه میکند، و این کار تا جایی که دکمه نگه داشته شود تکرار میشود.

9. حال که دیتا جمع آوری شده با فشردن دکمه `Train & Predict` تابع زیر اجرا میشود:

```JavaScript
async function trainAndPredict() {
    // تنظیم پرچم پیش‌بینی به "false" تا مطمئن شوید که در طول آموزش پیش‌بینی انجام نمی‌شود.
    predict = false;

    // مخلوط کردن داده‌های آموزش با تابع shuffleCombo
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

    // تبدیل خروجی‌ها به یک تنسور یک‌بعدی با نوع داده "int32"
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");

    // تولید خروجی‌ها به صورت one-hot encoding
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);

    // تبدیل ورودی‌ها به یک تنسور با استفاده از تابع stack
    let inputsAsTensor = tf.stack(trainingDataInputs);

    // آموزش مدل با استفاده از داده‌های آموزش و خروجی‌های one-hot
    let results = await model.fit(inputsAsTensor, oneHotOutputs, {
        shuffle: true,       // مخلوط کردن داده‌ها
        batchSize: 5,        // اندازه دسته آموزش
        epochs: 10,          // تعداد دوره‌های آموزش
        callbacks: {
            onEpochEnd: logProgress, // فراخوانی تابع logProgress در پایان هر دوره آموزش
        },
    });

    // آزاد کردن منابع مصرفی تنسورها
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();

    // تنظیم پرچم پیش‌بینی به "true" تا بتوانید با مدل پیش‌بینی انجام دهید.
    predict = true;
    predictLoop();
}

```

این تابع ابتدا داده‌های آموزش را مخلوط کرده و سپس آن‌ها را به تنسورهای TensorFlow.js تبدیل می‌کند. سپس مدل با استفاده از داده‌های آموزش و خروجی‌های one-hot تعیین شده، آموزش داده می‌شود. در پایان هر دوره آموزش، تابع `logProgress` به عنوان یک callback فراخوانی می‌شود. سپس تنسورها آزاد شده و پرچم پیش‌بینی به "true" تنظیم می‌شود تا بتوان با مدل پیش‌بینی انجام داد.

10. و در پایان تابع `predictLoop` اجرا میشود:

```JavaScript
function predictLoop() {
    // اگر پرچم پیش‌بینی "true" باشد، پیش‌بینی را انجام دهید.
    if (predict) {
        tf.tidy(function () {
            // محاسبه ویژگی‌های تصویر بر روی فریم فعلی
            let imageFeatures = calculateFeaturesOnCurrentFrame();

            // انجام پیش‌بینی با استفاده از مدل
            let prediction = model
                .predict(imageFeatures.expandDims())
                .squeeze();

            // یافتن شاخص بالاترین احتمال در پیش‌بینی
            let highestIndex = prediction.argMax().arraySync();

            // تبدیل نتایج پیش‌بینی به آرایه
            let predictionArray = prediction.arraySync();

            // نمایش نتیجه پیش‌بینی در صفحه نمایش
            STATUS.innerText =
                "Prediction: " +
                CLASS_NAMES[highestIndex] +
                " with " +
                Math.floor(predictionArray[highestIndex] * 100) +
                "% confidence";
        });

        // بازخوانی تابع با استفاده از requestAnimationFrame
        window.requestAnimationFrame(predictLoop);
    }
}
```

این تابع به طور مداوم عملکرد مدل را بر روی تصاویر جاری بررسی می‌کند و نتایج پیش‌بینی را به کاربر نمایش می‌دهد.

-   ابتدا بررسی می‌شود که آیا پرچم پیش‌بینی (`predict`) برابر با "true" است یا نه. اگر بله، پیش‌بینی انجام می‌شود.
-   با استفاده از `calculateFeaturesOnCurrentFrame()`، ویژگی‌های تصویر روی فریم فعلی محاسبه می‌شوند.
-   سپس با استفاده از مدل، پیش‌بینی بر روی ویژگی‌های تصویر محاسبه شده انجام می‌شود.
-   با استفاده از `argMax()`، شاخص با بیشترین احتمال در نتایج پیش‌بینی یافت می‌شود.
-   نتایج پیش‌بینی به آرایه تبدیل می‌شوند و سپس نتیجه پیش‌بینی و احتمال آن به صفحه نمایش نمایش داده می‌شود.
-   در نهایت، تابع `predictLoop` با استفاده از `requestAnimationFrame` بازخوانی می‌شود تا به صورت مداوم و در زمان واقعی پیش‌بینی‌ها انجام شود (با توجه به مقدار `predict`).
