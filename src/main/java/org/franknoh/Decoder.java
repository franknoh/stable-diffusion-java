package org.franknoh;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Paths;
import java.util.Random;

public class Decoder {
    private ZooModel decoder;
    private Device device;
    private NDManager manager;
    private Predictor<NDList, NDList> diffusion_predictor;
    Decoder() {
        if(Engine.getInstance().getGpuCount() > 0) {
            this.device = Device.gpu();
        } else {
            this.device = Device.cpu();
        }
        this.manager = NDManager.newBaseManager(this.device);
        Criteria<NDList, NDList> decoder_c = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(Paths.get("src/main/resources/decoder.pt"))
                .optEngine("PyTorch")
                .optDevice(device)
                .build();
        try {
            this.decoder = decoder_c.loadModel();
        } catch (IOException | ModelNotFoundException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
        this.diffusion_predictor = this.decoder.newPredictor();
    }
    public NDArray run(NDArray latent) {
        NDList decoder_input = new NDList(latent);
        NDList decoder_output;
        try {
            decoder_output = diffusion_predictor.predict(decoder_input);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        return decoder_output.get(0);
    }
    public void saveImage(NDArray t_latent, String name) {
        t_latent = t_latent.duplicate();
        NDArray t_image = this.run(t_latent).get(0);
        t_image = t_image.add(1).mul(127.5f).round().clip(0, 255).toType(DataType.UINT8, false).transpose(1, 2, 0);
        OutputStream t_stream = null;
        try {
            t_stream = new FileOutputStream(Paths.get("src/main/resources/out/"+name+".png").toFile());
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        try {
            ImageFactory.getInstance().fromNDArray(t_image).save(t_stream, "png");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
