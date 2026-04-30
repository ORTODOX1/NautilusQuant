`default_nettype none

// Top-level wrapper: stitches GU layers, PU, QU into a fused pipeline.
// Real ENC macro: LDV -> GU.L1 -> GU.L2 -> GU.L3 -> PU -> QU -> QJL -> PACK -> STV.
module nqx_top #(
    parameter int unsigned DATA_W = 32,
    parameter int unsigned DIM    = 128,
    parameter int unsigned BITS   = 3,
    parameter        string ROM_PATH = "golden_rom.mem"
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       in_valid,
    input  wire signed [DATA_W-1:0]   in_vec [0:DIM-1],
    output wire                       out_valid,
    output wire [BITS-1:0]            q_out  [0:DIM-1],
    output wire signed [DATA_W-1:0]   min_out,
    output wire signed [DATA_W-1:0]   max_out
);

    wire signed [DATA_W-1:0] gu_l1_out [0:DIM-1];
    wire signed [DATA_W-1:0] gu_l2_out [0:DIM-1];
    wire signed [DATA_W-1:0] gu_l3_out [0:DIM-1];
    wire signed [DATA_W-1:0] pu_r_out  [0:DIM-1];

    wire l1_valid, l2_valid, l3_valid, pu_valid;

    // GU layer instances are generated once per pair lane; for the skeleton we
    // pass-through to keep the structural diagram stable. Replace with the
    // per-lane givens_lane network during synthesis.
    assign l1_valid = in_valid;
    genvar gi;
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : gu_pass
            assign gu_l1_out[gi] = in_vec[gi];
            assign gu_l2_out[gi] = gu_l1_out[gi];
            assign gu_l3_out[gi] = gu_l2_out[gi];
        end
    endgenerate
    assign l2_valid = l1_valid;
    assign l3_valid = l2_valid;

    // Polar unit (per pair); skeleton wires through.
    assign pu_valid = l3_valid;
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : pu_pass
            assign pu_r_out[gi] = gu_l3_out[gi];
        end
    endgenerate

    quant_unit #(
        .DATA_W(DATA_W),
        .DIM   (DIM),
        .BITS  (BITS)
    ) u_qu (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (pu_valid),
        .in_vec   (pu_r_out),
        .out_valid(out_valid),
        .q_out    (q_out),
        .min_out  (min_out),
        .max_out  (max_out)
    );

endmodule

`default_nettype wire
