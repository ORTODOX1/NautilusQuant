`default_nettype none

module golden_rom #(
    parameter int unsigned DATA_W = 32,
    parameter int unsigned ADDR_W = 12,
    parameter int unsigned DEPTH  = 1024,
    parameter        string MEM_PATH = "golden_rom.mem"
) (
    input  wire                  clk,
    input  wire [ADDR_W-1:0]     addr,
    output reg  [DATA_W-1:0]     data
);

    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(MEM_PATH, mem);
    end

    always_ff @(posedge clk) begin
        data <= mem[addr];
    end

endmodule

`default_nettype wire
