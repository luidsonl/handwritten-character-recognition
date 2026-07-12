const tf = require('@tensorflow/tfjs');
const path = require('path');

async function test() {
  const modelPath = 'file://' + path.resolve('public/model/model.json');
  console.log('Loading from:', modelPath);
  
  try {
    const model = await tf.loadLayersModel(modelPath);
    console.log('Model loaded!');
    console.log('Weights:');
    model.weights.forEach(w => console.log('  ', w.name, w.shape));
  } catch(e) {
    console.error('ERROR:', e.message);
    
    // Try to build model from topology manually
    const fs = require('fs');
    const modelJson = JSON.parse(fs.readFileSync(path.resolve('public/model/model.json'), 'utf8'));
    console.log('\nTopology config layers:');
    modelJson.modelTopology.model_config.config.layers.forEach((l, i) => {
      console.log(`  ${i}: ${l.class_name} name=${l.config.name || 'MISSING'}`);
    });
    console.log('\nWeight manifest names:');
    modelJson.weightsManifest[0].weights.forEach(w => {
      console.log(`  ${w.name} ${w.shape}`);
    });
  }
}
test();
