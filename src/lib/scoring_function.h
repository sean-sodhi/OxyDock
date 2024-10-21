/*
   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Sean Sodhi
*/

#ifndef VINA_SCORING_FUNCTION_H
#define VINA_SCORING_FUNCTION_H

// Standard library includes
#include <stdlib.h>
#include <list>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <stdint.h>
#include <stdio.h>
// Include necessary headers
#include "atom.h"
#include "conf_independent.h"
#include "potentials.h"
#include "common.h"

// Include Treelite C API for XGBoost model integration
#include <treelite/c_api.h>

// Forward declaration of the 'model' struct
struct model;

// Enumeration of scoring function choices
enum scoring_function_choice {SF_VINA, SF_AD42, SF_VINARDO};

// The ScoringFunction class encapsulates the scoring function used in docking
class ScoringFunction {
public:
    // Default constructor
    ScoringFunction() : m_xgboost_model(nullptr) { }

    // Parameterized constructor
    // Initializes the scoring function based on the choice, weights, oxygen level, oxygen weight, and XGBoost model path
    ScoringFunction(const scoring_function_choice sf_choice, const flv& weights,
                    fl oxygen_level = 1.0, fl oxygen_weight = 0.0,
                    const std::string& xgboost_model_path = "") {
        // Set the oxygen level (between 0.0 and 1.0)
        m_oxygen_level = oxygen_level;

        // Store the weights provided
        m_weights = weights;

        // Initialize potentials and conf_independents based on the scoring function choice
        switch (sf_choice)
        {
            case SF_VINA:
            {
                // Vina potentials
                m_potentials.push_back(new vina_gaussian(0, 0.5, 8.0));
                m_potentials.push_back(new vina_gaussian(3, 2.0, 8.0));
                m_potentials.push_back(new vina_repulsion(0.0, 8.0));
                m_potentials.push_back(new vina_hydrophobic(0.5, 1.5, 8.0));
                m_potentials.push_back(new vina_non_dir_h_bond(-0.7, 0, 8.0));
                m_potentials.push_back(new linearattraction(20.0));

                // Vina conf_independents
                m_conf_independents.push_back(new num_tors_div());

                // Set atom typing and cutoffs
                m_atom_typing = atom_type::XS;
                m_cutoff = 8.0;
                m_max_cutoff = 20.0;
                break;
            }
            case SF_VINARDO:
            {
                // Vinardo potentials
                m_potentials.push_back(new vinardo_gaussian(0, 0.8, 8.0));
                m_potentials.push_back(new vinardo_repulsion(0, 8.0));
                m_potentials.push_back(new vinardo_hydrophobic(0, 2.5, 8.0));
                m_potentials.push_back(new vinardo_non_dir_h_bond(-0.6, 0, 8.0));
                m_potentials.push_back(new linearattraction(20.0));

                // Vinardo conf_independents
                m_conf_independents.push_back(new num_tors_div());

                // Set atom typing and cutoffs
                m_atom_typing = atom_type::XS;
                m_cutoff = 8.0;
                m_max_cutoff = 20.0;
                break;
            }
            case SF_AD42:
            {
                // AutoDock4.2 potentials
                m_potentials.push_back(new ad4_vdw(0.5, 100000, 8.0));
                m_potentials.push_back(new ad4_hb(0.5, 100000, 8.0));
                m_potentials.push_back(new ad4_electrostatic(100, 20.48));
                m_potentials.push_back(new ad4_solvation(3.6, 0.01097, true, 20.48));
                m_potentials.push_back(new linearattraction(20.0));

                // AutoDock4.2 conf_independents
                m_conf_independents.push_back(new ad4_tors_add());

                // Set atom typing and cutoffs
                m_atom_typing = atom_type::AD;
                m_cutoff = 20.48;
                m_max_cutoff = 20.48;
                break;
            }
            default:
            {
                // Invalid scoring function choice
                std::cout << "INSIDE ScoringFunction::ScoringFunction()   sf_choice = " << sf_choice << "\n";
                VINA_CHECK(false);
                break;
            }
        }

        // Add the OxygenPotential if an oxygen weight is provided
        if (oxygen_weight != 0.0) {
            // Create a new OxygenPotential with the specified oxygen level
            m_potentials.push_back(new OxygenPotential(m_oxygen_level));
            // Add the oxygen weight to the weights vector
            m_weights.push_back(oxygen_weight);
        }

        // Set the number of potentials and conf_independents
        m_num_potentials = m_potentials.size();
        m_num_conf_independents = m_conf_independents.size();

        // Load the XGBoost model if a valid path is provided
        if (!xgboost_model_path.empty()) {
            m_xgboost_model = new XGBoostScoringFunction(xgboost_model_path);
        } else {
            m_xgboost_model = nullptr; // No XGBoost model used
        }
    }

    // Destructor
    // Cleans up dynamically allocated memory
    ~ScoringFunction() {
        // Delete all potentials
        for (Potential* p : m_potentials) {
            delete p;
        }
        // Delete all conf_independents
        for (ConfIndependent* ci : m_conf_independents) {
            delete ci;
        }
        // Delete the XGBoost model if it was initialized
        if (m_xgboost_model) {
            delete m_xgboost_model;
        }
    }

    // Evaluate the interaction between two atoms given their types and distance
    fl eval(atom& a, atom& b, fl r) const {
        fl acc = 0; // Accumulator for the total energy

        // Loop over all potentials
        VINA_FOR (i, m_num_potentials)
        {
            // Accumulate the weighted energy contribution from each potential
            acc += m_weights[i] * m_potentials[i]->eval(a, b, r);
        }
        return acc; // Return the total energy
    }

    // Evaluate the interaction between two atom types given their types and distance
    fl eval(sz t1, sz t2, fl r) const {
        fl acc = 0; // Accumulator for the total energy

        // Loop over all potentials
        VINA_FOR (i, m_num_potentials)
        {
            // Accumulate the weighted energy contribution from each potential
            acc += m_weights[i] * m_potentials[i]->eval(t1, t2, r);
        }
        return acc; // Return the total energy
    }

    // Evaluate configuration-independent terms and apply XGBoost model if available
    fl conf_independent(const model& m, fl e) const {
        // Iterator for weights, starting after the potentials
        flv::const_iterator it = m_weights.begin() + m_num_potentials;

        // Create inputs for configuration-independent terms
        conf_independent_inputs in(m);

        // Loop over all conf_independents
        VINA_FOR (i, m_num_conf_independents)
        {
            // Evaluate each conf_independent term and update the energy 'e'
            e = m_conf_independents[i]->eval(in, e, it);
        }

        // Ensure the weights iterator has reached the end
        assert(it == m_weights.end());

        // If XGBoost model is available, use it to adjust the energy
        if (m_xgboost_model) {
            // Evaluate the XGBoost model with features extracted from 'm'
            fl xgboost_energy = eval_xgboost(m);
            // Add the XGBoost energy adjustment to 'e'
            e += xgboost_energy;
        }

        return e; // Return the total energy
    }

    // Getter methods for various properties
    fl get_cutoff() const { return m_cutoff; }
    fl get_max_cutoff() const { return m_max_cutoff; }
    atom_type::t get_atom_typing() const { return m_atom_typing; }

    // Get a vector of atom types based on the atom typing scheme
    szv get_atom_types() const {
        szv tmp;
        VINA_FOR(i, num_atom_types(m_atom_typing))
        {
            tmp.push_back(i);
        }
        return tmp;
    }

    // Get the number of atom types
    sz get_num_atom_types() const { return num_atom_types(m_atom_typing); }

    // Get the weights vector
    flv get_weights() const { return m_weights; }

private:
    // Member variables
    std::vector<Potential*> m_potentials;          // List of potentials used in the scoring function
    std::vector<ConfIndependent*> m_conf_independents; // List of configuration-independent terms
    flv m_weights;                                 // Weights corresponding to each potential and conf_independent
    fl m_cutoff;                                   // Cutoff distance for interactions
    fl m_max_cutoff;                               // Maximum cutoff distance
    int m_num_potentials;                          // Number of potentials
    int m_num_conf_independents;                   // Number of configuration-independent terms
    atom_type::t m_atom_typing;                    // Atom typing scheme used
    fl m_oxygen_level;                             // Oxygen level parameter (0.0 to 1.0)

    // XGBoostScoringFunction class encapsulates the XGBoost model integration
    class XGBoostScoringFunction {
    public:
        // Constructor: loads the XGBoost model using Treelite
        XGBoostScoringFunction(const std::string& model_path) {
            // Load the compiled Treelite model from the provided path
            int ret = TreelitePredictorLoad(model_path.c_str(), 1, &predictor_);
            if (ret != 0) {
                throw std::runtime_error("Failed to load XGBoost model.");
            }
        }

        // Destructor: frees the predictor resources
        ~XGBoostScoringFunction() {
            // Free the predictor handle
            TreelitePredictorFree(predictor_);
        }

        // Predict method: uses the XGBoost model to predict binding affinity based on features
        fl predict(const std::vector<float>& features) const {
            size_t out_result_size;
            const float* out_result;

            // Create a PredictorBatch structure to hold the features
            TreelitePredictorEntry data_entry;
            data_entry.data = features.data();
            data_entry.num_row = 1;
            data_entry.num_col = features.size();

            // Perform prediction
            int ret = TreelitePredictorPredictBatch(predictor_, &data_entry, 0, 1, &out_result_size, &out_result);
            if (ret != 0) {
                throw std::runtime_error("Failed to predict using XGBoost model.");
            }
            return static_cast<fl>(out_result[0]); // Return the predicted binding affinity
        }

    private:
        PredictorHandle predictor_; // Handle to the Treelite predictor
    };

    XGBoostScoringFunction* m_xgboost_model; // Pointer to the XGBoostScoringFunction instance

    // Method to evaluate the XGBoost model and get the energy adjustment
    fl eval_xgboost(const model& m) const {
        // Extract features from the model 'm'
        std::vector<float> features = extract_features(m);
        // Use the XGBoost model to predict the binding affinity
        return m_xgboost_model->predict(features);
    }

    // Method to extract features from the model for XGBoost prediction
    std::vector<float> extract_features(const model& m) const {
        std::vector<float> features;

        // Example feature extraction:

        // 1. Oxygen level (normalized between 0.0 and 1.0)
        features.push_back(static_cast<float>(m_oxygen_level));

        // 2. Number of torsions in the ligand
        features.push_back(static_cast<float>(m.get_size().num_torsions));

        // 3. Total intermolecular energy (without oxygen potential)
        // For demonstration purposes, we assume a method exists:
        // fl intermolecular_energy = m.calculate_intermolecular_energy();
        // features.push_back(static_cast<float>(intermolecular_energy));

        // 4. Other features as needed, e.g., hydrophobic surface area, number of hydrogen bonds, etc.
        // Ensure that the features extracted here match those used during model training

        return features; // Return the vector of features
    }
};

// OxygenPotential class definition
// This class defines a potential that penalizes interactions under hypoxic conditions
class OxygenPotential : public Potential {
public:
    // Constructor: sets the oxygen level
    OxygenPotential(fl oxygen_level) : m_oxygen_level(oxygen_level) {}

    // Evaluate the potential between two atoms 'a' and 'b' at distance 'r'
    fl eval(const atom& a, const atom& b, fl r) const override {
        // Penalty increases as oxygen level decreases (hypoxia)
        // The penalty decreases exponentially with distance 'r'
        const fl lambda = 1.0; // Decay constant, adjust based on empirical data
        fl penalty = (1.0 - m_oxygen_level) * std::exp(-lambda * r);
        return penalty; // Return the calculated penalty
    }

    // Evaluate the potential between two atom types given their types and distance
    fl eval(sz t1, sz t2, fl r) const override {
        // Implement if necessary for atom typing; otherwise, return 0
        return 0.0; // No penalty based solely on atom types
    }

private:
    fl m_oxygen_level; // Oxygen level parameter (0.0 for hypoxic, 1.0 for normoxic)
};

#endif // VINA_SCORING_FUNCTION_H
