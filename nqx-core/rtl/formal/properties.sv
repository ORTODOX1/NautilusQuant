`default_nettype none

// Formal properties for NQX-Core. SymbiYosys (sby) drives BMC / k-induction
// against this module via formal/orthogonality.sby.

module properties #(
    parameter int unsigned DATA_W = 32,
    parameter int unsigned DIM    = 16
) (
    input wire                       clk,
    input wire                       rst_n,
    input wire                       in_valid,
    input wire signed [DATA_W-1:0]   in_a,
    input wire signed [DATA_W-1:0]   in_b,
    input wire signed [DATA_W-1:0]   cos_w,
    input wire signed [DATA_W-1:0]   sin_w,
    output wire                      out_valid,
    output wire signed [DATA_W-1:0]  out_a,
    output wire signed [DATA_W-1:0]  out_b
);

    givens_lane #(.DATA_W(DATA_W)) u_lane (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_a     (in_a),
        .in_b     (in_b),
        .in_cos   (cos_w),
        .in_sin   (sin_w),
        .out_valid(out_valid),
        .out_a    (out_a),
        .out_b    (out_b)
    );

    // P1: rst_n low forces out_valid low.
    p_reset_clears_valid: assert property (
        @(posedge clk) (!rst_n) |=> (!out_valid)
    );

    // P2: in_valid pulse propagates to out_valid within 2 clocks.
    p_valid_propagates: assert property (
        @(posedge clk) disable iff (!rst_n)
        in_valid |-> ##2 out_valid
    );

    // P3: when (cos_w, sin_w) = (1, 0) and inputs are bounded, the rotation
    // is the identity — out_a == in_a (delayed by 2 cycles).
    p_identity_rotation: assert property (
        @(posedge clk) disable iff (!rst_n)
        (in_valid && cos_w == 32'sd1 && sin_w == 32'sd0
         && (in_a >>> 16) == 0 && (in_b >>> 16) == 0)
        |-> ##2 (out_a == ($past(in_a, 2)))
    );

    // P4: pair non-overlap — golden_rom never returns the same index for i and j.
    // Encoded as a structural invariant (placeholder; the full proof is in
    // GoldenAngleLUT regression in tests/test_lut.py).

endmodule

`default_nettype wire
