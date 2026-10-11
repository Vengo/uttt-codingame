//! Pure Rust INT8 Inference Module for Ultimate Tic-Tac-Toe
//!
//! Provides zero-dependency, fixed-point INT8 forward evaluation for:
//! - Conv2d 3x3 (with padding 1 and HWC layout)
//! - Residual blocks (with integer skip connection addition)
//! - Policy & Value dual-heads (1x1 convs and linear projections)
//! - Feature tensor extraction from `UltimateTicTacToeBoard`
//! - Embedded binary weights (< 320 KB) compatible with CodinGame single-file submission.

use crate::uttt_trait_solving::UltimateTicTacToeBoard;

pub static DEFAULT_MODEL_BYTES: &[u8] = include_bytes!("model_int8.bin");

/// 5x9x9 input board features in HWC format: [row][col][channel]
pub type BoardTensor = [[[i8; 5]; 9]; 9];

/// Extract the 5x9x9 spatial feature tensor and legal move mask from an `UltimateTicTacToeBoard`.
///
/// Channels:
///   0: Active player marks
///   1: Opponent marks
///   2: Active player won subboards (across all 9 cells)
///   3: Opponent won subboards (across all 9 cells)
///   4: Legal target subboard mask (across all 9 cells of legal subboards)
pub fn extract_features(board: &UltimateTicTacToeBoard) -> (BoardTensor, [bool; 81]) {
    let mut tensor: BoardTensor = [[[0; 5]; 9]; 9];
    let mut legal_mask = [false; 81];

    let (active_bb, opp_bb) = if board.x_to_move {
        (board.xbb, board.obb)
    } else {
        (board.obb, board.xbb)
    };

    let (active_supers, opp_supers) = if board.x_to_move {
        (board.x_supers, board.o_supers)
    } else {
        (board.o_supers, board.x_supers)
    };

    let tied_mask = board.get_tied_mask();
    let complete_bbs = board.x_supers | board.o_supers | tied_mask;

    let target_mask: u16 = if board.last_super_index >= 9 || (complete_bbs & (1 << board.last_super_index)) != 0 {
        (!complete_bbs) & 0x1FF
    } else {
        1 << board.last_super_index
    };

    for sb in 0..9usize {
        let sb_r = sb / 3;
        let sb_c = sb % 3;
        let is_target = (target_mask & (1 << sb)) != 0;
        let is_active_won = (active_supers & (1 << sb)) != 0;
        let is_opp_won = (opp_supers & (1 << sb)) != 0;
        let occ = board.xbb[sb] | board.obb[sb];

        for sq in 0..9usize {
            let sq_r = sq / 3;
            let sq_c = sq % 3;
            let r = sb_r * 3 + sq_r;
            let c = sb_c * 3 + sq_c;
            let move_idx = 9 * sb + sq;

            if (active_bb[sb] & (1 << sq)) != 0 {
                tensor[r][c][0] = 1;
            }
            if (opp_bb[sb] & (1 << sq)) != 0 {
                tensor[r][c][1] = 1;
            }
            if is_active_won {
                tensor[r][c][2] = 1;
            }
            if is_opp_won {
                tensor[r][c][3] = 1;
            }
            if is_target {
                tensor[r][c][4] = 1;
            }

            if is_target && (occ & (1 << sq)) == 0 {
                legal_mask[move_idx] = true;
            }
        }
    }

    (tensor, legal_mask)
}

#[derive(Clone, Debug)]
pub struct Conv3x3Layer {
    pub in_channels: usize,
    pub out_channels: usize,
    /// Weights shape: [out_channels, 3, 3, in_channels]
    pub weights: Vec<i8>,
    pub bias: Vec<i32>,
    pub mult: f32,
}

#[derive(Clone, Debug)]
pub struct ResBlockLayers {
    pub conv1: Conv3x3Layer,
    pub conv2: Conv3x3Layer,
}

#[derive(Clone, Debug)]
pub struct Conv1x1Layer {
    pub in_channels: usize,
    pub out_channels: usize,
    /// Weights shape: [out_channels, in_channels]
    pub weights: Vec<i8>,
    pub bias: Vec<i32>,
    pub mult: f32,
}

#[derive(Clone, Debug)]
pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,
    /// Weights shape: [out_features, in_features]
    pub weights: Vec<i8>,
    pub bias: Vec<i32>,
    pub mult: f32,
}

#[derive(Clone, Debug)]
pub struct ModelInt8 {
    pub num_channels: usize,
    pub num_res_blocks: usize,
    pub conv0: Conv3x3Layer,
    pub res_blocks: Vec<ResBlockLayers>,
    pub conv_p: Conv1x1Layer,
    pub fc_p: LinearLayer,
    pub conv_v: Conv1x1Layer,
    pub fc_v1: LinearLayer,
    pub fc_v2: LinearLayer,
}

/// Scratch workspace for zero-allocation inference.
pub struct InferenceWorkspace {
    padded: Box<[[[i8; 64]; 11]; 11]>,
    buf_a: Box<[[[i8; 64]; 9]; 9]>,
    buf_b: Box<[[[i8; 64]; 9]; 9]>,
    buf_c: Box<[[[i8; 64]; 9]; 9]>,
}

impl Default for InferenceWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceWorkspace {
    pub fn new() -> Self {
        InferenceWorkspace {
            padded: Box::new([[[0i8; 64]; 11]; 11]),
            buf_a: Box::new([[[0i8; 64]; 9]; 9]),
            buf_b: Box::new([[[0i8; 64]; 9]; 9]),
            buf_c: Box::new([[[0i8; 64]; 9]; 9]),
        }
    }
}

impl ModelInt8 {
    /// Load default model embedded in the binary.
    pub fn default_model() -> Self {
        Self::from_bytes(DEFAULT_MODEL_BYTES).expect("Failed to load embedded default INT8 model")
    }

    /// Dynamically load INT8 model from an external file on disk at runtime.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read INT8 model file: {}", e))?;
        Self::from_bytes(&bytes).map_err(|e| e.to_string())
    }

    /// Deserialize model from a binary buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 16 {
            return Err("Model binary buffer too small");
        }
        if &bytes[0..8] != b"UTTTNN01" {
            return Err("Invalid magic bytes in model header");
        }

        let num_channels = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let num_res_blocks = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

        let mut offset = 16;

        // Helper to read conv3x3
        let read_conv3x3 = |offset: &mut usize, in_c: usize, out_c: usize| -> Result<Conv3x3Layer, &'static str> {
            let w_len = out_c * 3 * 3 * in_c;
            if *offset + w_len > bytes.len() {
                return Err("EOF reading weights");
            }
            let weights = bytes[*offset..*offset + w_len]
                .iter()
                .map(|&b| b as i8)
                .collect();
            *offset += w_len;

            let b_len = out_c * 4;
            if *offset + b_len > bytes.len() {
                return Err("EOF reading bias");
            }
            let mut bias = Vec::with_capacity(out_c);
            for i in 0..out_c {
                let start = *offset + i * 4;
                let val = i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
                bias.push(val);
            }
            *offset += b_len;

            if *offset + 4 > bytes.len() {
                return Err("EOF reading mult");
            }
            let mult = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;

            Ok(Conv3x3Layer {
                in_channels: in_c,
                out_channels: out_c,
                weights,
                bias,
                mult,
            })
        };

        // Helper to read conv1x1
        let read_conv1x1 = |offset: &mut usize, in_c: usize, out_c: usize| -> Result<Conv1x1Layer, &'static str> {
            let w_len = out_c * in_c;
            if *offset + w_len > bytes.len() {
                return Err("EOF reading 1x1 weights");
            }
            let weights = bytes[*offset..*offset + w_len]
                .iter()
                .map(|&b| b as i8)
                .collect();
            *offset += w_len;

            let b_len = out_c * 4;
            if *offset + b_len > bytes.len() {
                return Err("EOF reading 1x1 bias");
            }
            let mut bias = Vec::with_capacity(out_c);
            for i in 0..out_c {
                let start = *offset + i * 4;
                let val = i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
                bias.push(val);
            }
            *offset += b_len;

            if *offset + 4 > bytes.len() {
                return Err("EOF reading 1x1 mult");
            }
            let mult = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;

            Ok(Conv1x1Layer {
                in_channels: in_c,
                out_channels: out_c,
                weights,
                bias,
                mult,
            })
        };

        // Helper to read linear
        let read_linear = |offset: &mut usize, in_f: usize, out_f: usize| -> Result<LinearLayer, &'static str> {
            let w_len = out_f * in_f;
            if *offset + w_len > bytes.len() {
                return Err("EOF reading linear weights");
            }
            let weights = bytes[*offset..*offset + w_len]
                .iter()
                .map(|&b| b as i8)
                .collect();
            *offset += w_len;

            let b_len = out_f * 4;
            if *offset + b_len > bytes.len() {
                return Err("EOF reading linear bias");
            }
            let mut bias = Vec::with_capacity(out_f);
            for i in 0..out_f {
                let start = *offset + i * 4;
                let val = i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
                bias.push(val);
            }
            *offset += b_len;

            if *offset + 4 > bytes.len() {
                return Err("EOF reading linear mult");
            }
            let mult = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;

            Ok(LinearLayer {
                in_features: in_f,
                out_features: out_f,
                weights,
                bias,
                mult,
            })
        };

        // 1. conv0 (5 -> num_channels)
        let conv0 = read_conv3x3(&mut offset, 5, num_channels)?;

        // 2. ResBlocks
        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for _ in 0..num_res_blocks {
            let conv1 = read_conv3x3(&mut offset, num_channels, num_channels)?;
            let conv2 = read_conv3x3(&mut offset, num_channels, num_channels)?;
            res_blocks.push(ResBlockLayers { conv1, conv2 });
        }

        // 3. Policy Head: conv_p (num_channels -> 2), fc_p (162 -> 81)
        let conv_p = read_conv1x1(&mut offset, num_channels, 2)?;
        let fc_p = read_linear(&mut offset, 2 * 9 * 9, 81)?;

        // 4. Value Head: conv_v (num_channels -> 1), fc_v1 (81 -> 64), fc_v2 (64 -> 1)
        let conv_v = read_conv1x1(&mut offset, num_channels, 1)?;
        let fc_v1 = read_linear(&mut offset, 81, 64)?;
        let fc_v2 = read_linear(&mut offset, 64, 1)?;

        Ok(ModelInt8 {
            num_channels,
            num_res_blocks,
            conv0,
            res_blocks,
            conv_p,
            fc_p,
            conv_v,
            fc_v1,
            fc_v2,
        })
    }

    /// Run full forward inference on the board features.
    /// Returns:
    ///   - policy_logits: [f32; 81] (illegal moves set to -1e9)
    ///   - value: f32 in [-1.0, 1.0] from active player's perspective
    pub fn forward(
        &self,
        input: &BoardTensor,
        legal_mask: &[bool; 81],
        ws: &mut InferenceWorkspace,
    ) -> ([f32; 81], f32) {
        let c = self.num_channels;

        // --- Layer 0: Conv0 (in=5, out=64, k=3, pad=1) + ReLU ---
        // Pad input (5 channels)
        for r in 0..11 {
            for col in 0..11 {
                for ch in 0..5 {
                    ws.padded[r][col][ch] = 0;
                }
            }
        }
        for r in 0..9 {
            for col in 0..9 {
                for ch in 0..5 {
                    ws.padded[r + 1][col + 1][ch] = input[r][col][ch];
                }
            }
        }

#[inline(always)]
fn dot_product_64(a: &[i8; 64], b: &[i8]) -> i32 {
    let mut acc0 = 0i32;
    let mut acc1 = 0i32;
    let mut acc2 = 0i32;
    let mut acc3 = 0i32;
    for i in 0..16 {
        acc0 += (a[i] as i32) * (b[i] as i32);
        acc1 += (a[i + 16] as i32) * (b[i + 16] as i32);
        acc2 += (a[i + 32] as i32) * (b[i + 32] as i32);
        acc3 += (a[i + 48] as i32) * (b[i + 48] as i32);
    }
    (acc0 + acc1) + (acc2 + acc3)
}

        for r in 0..9 {
            for col in 0..9 {
                for oc in 0..c {
                    let mut acc = self.conv0.bias[oc];
                    let w_oc = &self.conv0.weights[oc * 45..(oc + 1) * 45];
                    let mut w_idx = 0;
                    for kh in 0..3 {
                        let in_r = &ws.padded[r + kh];
                        for kw in 0..3 {
                            let in_cell = &in_r[col + kw];
                            for ic in 0..5 {
                                acc += (in_cell[ic] as i32) * (w_oc[w_idx] as i32);
                                w_idx += 1;
                            }
                        }
                    }
                    let scaled = ((acc as f32) * self.conv0.mult).round() as i32;
                    ws.buf_a[r][col][oc] = scaled.clamp(0, 127) as i8;
                }
            }
        }

        // --- Residual Blocks ---
        for block in &self.res_blocks {
            // Conv1 (in=64, out=64, k=3, pad=1) + ReLU
            // Fast border zeroing
            for col in 0..11 {
                ws.padded[0][col] = [0; 64];
                ws.padded[10][col] = [0; 64];
            }
            for r in 1..10 {
                ws.padded[r][0] = [0; 64];
                ws.padded[r][10] = [0; 64];
            }
            for r in 0..9 {
                for col in 0..9 {
                    ws.padded[r + 1][col + 1] = ws.buf_a[r][col];
                }
            }

            let w_stride = 9 * c;
            for r in 0..9 {
                for col in 0..9 {
                    for oc in 0..c {
                        let mut acc = block.conv1.bias[oc];
                        let w_oc = &block.conv1.weights[oc * w_stride..(oc + 1) * w_stride];
                        let mut k_offset = 0;
                        for kh in 0..3 {
                            let in_r = &ws.padded[r + kh];
                            for kw in 0..3 {
                                acc += dot_product_64(&in_r[col + kw], &w_oc[k_offset..k_offset + 64]);
                                k_offset += 64;
                            }
                        }
                        let scaled = ((acc as f32) * block.conv1.mult).round() as i32;
                        ws.buf_b[r][col][oc] = scaled.clamp(0, 127) as i8;
                    }
                }
            }

            // Conv2 (in=64, out=64, k=3, pad=1) (no ReLU before skip)
            for col in 0..11 {
                ws.padded[0][col] = [0; 64];
                ws.padded[10][col] = [0; 64];
            }
            for r in 1..10 {
                ws.padded[r][0] = [0; 64];
                ws.padded[r][10] = [0; 64];
            }
            for r in 0..9 {
                for col in 0..9 {
                    ws.padded[r + 1][col + 1] = ws.buf_b[r][col];
                }
            }

            for r in 0..9 {
                for col in 0..9 {
                    for oc in 0..c {
                        let mut acc = block.conv2.bias[oc];
                        let w_oc = &block.conv2.weights[oc * w_stride..(oc + 1) * w_stride];
                        let mut k_offset = 0;
                        for kh in 0..3 {
                            let in_r = &ws.padded[r + kh];
                            for kw in 0..3 {
                                acc += dot_product_64(&in_r[col + kw], &w_oc[k_offset..k_offset + 64]);
                                k_offset += 64;
                            }
                        }
                        let scaled = ((acc as f32) * block.conv2.mult).round() as i32;
                        ws.buf_c[r][col][oc] = scaled.clamp(-128, 127) as i8;
                    }
                }
            }

            // Skip addition + ReLU: buf_a = clamp(buf_a + buf_c, 0, 127)
            for r in 0..9 {
                for col in 0..9 {
                    for oc in 0..c {
                        let sum = (ws.buf_a[r][col][oc] as i32) + (ws.buf_c[r][col][oc] as i32);
                        ws.buf_a[r][col][oc] = sum.clamp(0, 127) as i8;
                    }
                }
            }
        }

        // --- Policy Head ---
        // Conv_p (64 -> 2, 1x1) + ReLU
        let mut flat_p = [0i8; 162];
        for r in 0..9 {
            for col in 0..9 {
                for oc in 0..2 {
                    let mut acc = self.conv_p.bias[oc];
                    let w_oc = &self.conv_p.weights[oc * c..(oc + 1) * c];
                    let in_cell = &ws.buf_a[r][col];
                    for ic in 0..c {
                        acc += (in_cell[ic] as i32) * (w_oc[ic] as i32);
                    }
                    let scaled = ((acc as f32) * self.conv_p.mult).round() as i32;
                    let val = scaled.clamp(0, 127) as i8;
                    // Flatten in PyTorch NCHW order: [channel * 81 + r * 9 + col]
                    flat_p[oc * 81 + r * 9 + col] = val;
                }
            }
        }

        // Linear_p: 162 -> 81 logits
        let mut policy_logits = [0.0f32; 81];
        for m in 0..81 {
            let mut acc = self.fc_p.bias[m];
            let w_m = &self.fc_p.weights[m * 162..(m + 1) * 162];
            for i in 0..162 {
                acc += (flat_p[i] as i32) * (w_m[i] as i32);
            }
            let mut logit = (acc as f32) * self.fc_p.mult;
            if !legal_mask[m] {
                logit = -1e9;
            }
            policy_logits[m] = logit;
        }

        // --- Value Head ---
        // Conv_v (64 -> 1, 1x1) + ReLU
        let mut flat_v = [0i8; 81];
        for r in 0..9 {
            for col in 0..9 {
                let mut acc = self.conv_v.bias[0];
                let in_cell = &ws.buf_a[r][col];
                for ic in 0..c {
                    acc += (in_cell[ic] as i32) * (self.conv_v.weights[ic] as i32);
                }
                let scaled = ((acc as f32) * self.conv_v.mult).round() as i32;
                flat_v[r * 9 + col] = scaled.clamp(0, 127) as i8;
            }
        }

        // Linear_v1: 81 -> 64 + ReLU
        let mut hidden_v = [0i8; 64];
        for o in 0..64 {
            let mut acc = self.fc_v1.bias[o];
            let w_o = &self.fc_v1.weights[o * 81..(o + 1) * 81];
            for i in 0..81 {
                acc += (flat_v[i] as i32) * (w_o[i] as i32);
            }
            let scaled = ((acc as f32) * self.fc_v1.mult).round() as i32;
            hidden_v[o] = scaled.clamp(0, 127) as i8;
        }

        // Linear_v2: 64 -> 1 + Tanh
        let mut acc_v2 = self.fc_v2.bias[0];
        for i in 0..64 {
            acc_v2 += (hidden_v[i] as i32) * (self.fc_v2.weights[i] as i32);
        }
        let raw_val = (acc_v2 as f32) * self.fc_v2.mult;
        let value = raw_val.tanh();

        (policy_logits, value)
    }

    /// Direct evaluation convenience method taking an `UltimateTicTacToeBoard`.
    pub fn evaluate(
        &self,
        board: &UltimateTicTacToeBoard,
        ws: &mut InferenceWorkspace,
    ) -> ([f32; 81], f32) {
        let (tensor, legal_mask) = extract_features(board);
        self.forward(&tensor, &legal_mask, ws)
    }
}
