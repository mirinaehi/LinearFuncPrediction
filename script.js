// 학습데이터 수집 및 손실데이터 정리
async function getData() {
  let rawData = []
  const number = 25
  for(let n=1; n<=number; n++)
  {
    let obj = {
      "x_axis" : n,
      "y_axis" : 2 * n + 1
    }
    rawData.push(obj)
  }
  // json 처리시
  // const dataResponse = await fetch('data.json');
  // const rawData = await dataResponse.json();

  const cleaned = rawData.map(data => ({
    x: data.x_axis,
    y: data.y_axis,
  }))
    .filter(data => (data.y != null && data.x != null));

  return cleaned;
}

/* 모델 arch 정의
   두 개의 layer(입력, 출력)를 사용
 */
function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

/**
 * 머신러닝에 사용할 수 있도록 data를 tensor로 변환한다.
 * 데이터 shuffling과 nomalization(정규화)를 진행
 */
function convertToTensor(data) {
  // 계산을 깔끔이 정돈하면 중간 tensor들을 dispose 할 수 있다.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.x)
    const labels = data.map(d => d.y);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

async function trainModel(model, inputs, labels, epochs) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;

  // 학습 루프 시작
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

// 예측실행
function testModel(model, inputData, normalizationData, epochs) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const num = 100
    const xs = tf.linspace(0, 1, num);
    const preds = model.predict(xs.reshape([num, 1]));

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.x, y: d.y,
  }));

  tfvis.render.scatterplot(
    {name: `모델 예측과 원본 데이터의 비교 epoch:${epochs}`},
    {values: [predictedPoints, originalPoints], series: ['예측', '원본']},
    {
      xLabel: 'x축',
      yLabel: 'y축',
      height: 300
    }
  );
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.x,
    y: d.y,
  }));

  tfvis.render.scatterplot(
    {name: 'y = 2x+1 그래프에 맞춰 점찍기(원본 데이터)'},
    {values},
    {
      xLabel: 'x축',
      yLabel: 'y축',
      height: 300
    }
  );

  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  // 학습할 수 있는 형태로 데이터 convert
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // 모델 학습
  await trainModel(model, inputs, labels, 100);

  // 데이터예측
  testModel(model, data, tensorData, 100);

  await trainModel(model, inputs, labels, 500);
  testModel(model, data, tensorData, 500);

  await trainModel(model, inputs, labels, 2000);
  testModel(model, data, tensorData, 2000);
}

document.addEventListener('DOMContentLoaded', run);
