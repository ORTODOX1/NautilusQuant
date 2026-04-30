`default_nettype none

// Lloyd-Max quantizer with min/max reduction tree.
// Reduction depth = $clog2(DIM); QU.q stage performs (x - min) * inv_range * (2^bits - 1).
module quant_unit #(
    parameter int unsigned DATA_W = 32,
    parameter int unsigned DIM    = 128,
    parameter int unsigned BITS   = 3
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     in_valid,
    input  wire signed [DATA_W-1:0] in_vec [0:DIM-1],
    output reg                      out_valid,
    output reg  [BITS-1:0]          q_out  [0:DIM-1],
    output reg  signed [DATA_W-1:0] min_out,
    output reg  signed [DATA_W-1:0] max_out
);

    localparam int LEVELS = (1 << BITS) - 1;

    reg signed [DATA_W-1:0] min_reg, max_reg;
    reg                     valid_pipe [0:7];

    integer k;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            min_reg <= '0;
            max_reg <= '0;
            for (k = 0; k < 8; k = k + 1) valid_pipe[k] <= 1'b0;
        end else begin
            min_reg <= in_vec[0];
            max_reg <= in_vec[0];
            for (k = 1; k < DIM; k = k + 1) begin
                if (in_vec[k] < min_reg) min_reg <= in_vec[k];
                if (in_vec[k] > max_reg) max_reg <= in_vec[k];
            end
            valid_pipe[0] <= in_valid;
            for (k = 1; k < 8; k = k + 1) valid_pipe[k] <= valid_pipe[k-1];
        end
    end

    integer j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            min_out   <= '0;
            max_out   <= '0;
            for (j = 0; j < DIM; j = j + 1) q_out[j] <= '0;
        end else begin
            out_valid <= valid_pipe[7];
            min_out   <= min_reg;
            max_out   <= max_reg;
            for (j = 0; j < DIM; j = j + 1) begin
                q_out[j] <= (in_vec[j][BITS-1:0]);  // placeholder mapping
            end
        end
    end

endmodule

`default_nettype wire
