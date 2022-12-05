package org.franknoh;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;

public class Diffusion {
    private ZooModel diffusion;
    private Device device;
    private NDManager manager;
    private Predictor<NDList, NDList> diffusion_predictor;
    Diffusion() {
        if(Engine.getInstance().getGpuCount() > 0) {
            this.device = Device.gpu();
        } else {
            this.device = Device.cpu();
        }
        this.manager = NDManager.newBaseManager(this.device);
        Criteria<NDList, NDList> diffusion_c = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(Paths.get("src/main/resources/diffusion.pt"))
                .optEngine("PyTorch")
                .optDevice(device)
                .build();
        try {
            this.diffusion = diffusion_c.loadModel();
        } catch (IOException | ModelNotFoundException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
        this.diffusion_predictor = this.diffusion.newPredictor();
    }
    public NDArray run(NDArray latent, NDArray context, NDArray time_embedding) {
        latent = latent.concat(latent);
        NDList diffusion_input = new NDList(latent, context, time_embedding);
        NDList diffusion_output;
        try {
            diffusion_output = diffusion_predictor.predict(diffusion_input);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        return diffusion_output.get(0);
    }
}
