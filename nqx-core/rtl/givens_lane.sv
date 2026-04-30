`default_nettype none

module givens_lane #(
    parameter int unsigned DATA_W = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     in_valid,
    input  wire signed [DATA_W-1:0] in_a,
    input  wire signed [DATA_W-1:0] in_b,
    input  wire signed [DATA_W-1:0] in_cos,
    input  wire signed [DATA_W-1:0] in_sin,
    output reg                      out_valid,
    output reg  signed [DATA_W-1:0] out_a,
    output reg  signed [DATA_W-1:0] out_b
);

    reg signed [DATA_W-1:0] s1_a, s1_b;
    reg signed [DATA_W-1:0] s1_cos, s1_sin;
    reg                     s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
        end else begin
            s1_valid <= in_valid;
            s1_a     <= in_a;
            s1_b     <= in_b;
            s1_cos   <= in_cos;
            s1_sin   <= in_sin;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
        end else begin
            out_valid <= s1_valid;
            out_a <= (s1_a * s1_cos) - (s1_b * s1_sin);
            out_b <= (s1_a * s1_sin) + (s1_b * s1_cos);
        end
    end

endmodule

`default_nettype wire
