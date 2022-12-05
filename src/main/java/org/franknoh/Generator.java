package org.franknoh;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class Generator {
    private final NDManager manager;
    Generator() {
        Device device;
        if(Engine.getInstance().getGpuCount() > 0) {
            device = Device.gpu();
        } else {
            device = Device.cpu();
        }
        this.manager = NDManager.newBaseManager(device);
    }

    NDArray sample(Shape shape) {
        return this.manager.randomNormal(shape, DataType.FLOAT32);
    }
}
