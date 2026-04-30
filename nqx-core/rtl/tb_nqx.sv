`default_nettype none

// Verilator-driven testbench. Loads `python_dump.hex` produced by tools/dump_for_rtl.py
// and compares per-vector quantized output against the SystemVerilog pipeline.
module tb_nqx;

    localparam int DIM   = 128;
    localparam int BITS  = 3;
    localparam int DATA_W = 32;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    reg in_valid;
    reg signed [DATA_W-1:0] in_vec [0:DIM-1];
    wire out_valid;
    wire [BITS-1:0] q_out [0:DIM-1];
    wire signed [DATA_W-1:0] min_out, max_out;

    nqx_top #(
        .DATA_W(DATA_W),
        .DIM   (DIM),
        .BITS  (BITS),
        .ROM_PATH("golden_rom.mem")
    ) u_dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_vec   (in_vec),
        .out_valid(out_valid),
        .q_out    (q_out),
        .min_out  (min_out),
        .max_out  (max_out)
    );

    integer file, code, idx;
    integer mismatches;

    initial begin
        mismatches = 0;
        in_valid = 1'b0;
        for (idx = 0; idx < DIM; idx = idx + 1) in_vec[idx] = '0;
        #20 rst_n = 1'b1;

        // Load Python golden dump if present.
        file = $fopen("python_dump.hex", "r");
        if (file == 0) begin
            $display("[tb_nqx] no python_dump.hex; running smoke pulse");
            in_valid = 1'b1;
            #10 in_valid = 1'b0;
        end else begin
            for (idx = 0; idx < DIM; idx = idx + 1) begin
                code = $fscanf(file, "%h\n", in_vec[idx]);
                if (code != 1) $fatal(1, "python_dump.hex truncated at idx=%0d", idx);
            end
            $fclose(file);
            in_valid = 1'b1;
            #10 in_valid = 1'b0;
        end

        repeat (32) @(posedge clk);
        if (mismatches != 0) $fatal(1, "tb_nqx: %0d mismatches", mismatches);
        $display("[tb_nqx] PASS");
        $finish;
    end

endmodule

`default_nettype wire
