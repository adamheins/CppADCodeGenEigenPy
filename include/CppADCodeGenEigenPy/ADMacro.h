#pragma once

#define BEGIN_AD_MODEL(Model, Scalar)              \
    struct Model : ADModel<Scalar> {               \
        Model(size_t input_dim, size_t output_dim) \
            : ADModel("Model", input_dim, output_dim){};

#define END_AD_MODEL };
