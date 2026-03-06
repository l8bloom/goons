package services

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/jupiterrider/ffi"
	"golang.org/x/sys/unix"
)

// opaque pointers
type (
	Context uintptr
)

type ContextParams struct {
	ModelPath                   *byte             // const char* model_path;
	ClipLPath                   *byte             // const char* clip_l_path;
	ClipGPath                   *byte             // const char* clip_g_path;
	ClipVisionPath              *byte             // const char* clip_vision_path;
	T5XXLPath                   *byte             // const char* t5xxl_path;
	LLMPath                     *byte             // const char* llm_path;
	LLMVisionPath               *byte             // const char* llm_vision_path;
	DiffusionModelPath          *byte             // const char* diffusion_model_path;
	HighNoiseDiffusionModelPath *byte             // const char* high_noise_diffusion_model_path;
	VAEPath                     *byte             // const char* vae_path;
	TAESDPath                   *byte             // const char* taesd_path;
	ControlNetPath              *byte             // const char* control_net_path;
	Embeddings                  *Embedding        // const sd_embedding_t* embeddings;
	EmbeddingCount              uint32            // uint32_t embedding_count;
	PhotoMakerPath              *byte             // const char* photo_maker_path;
	TensorTypeRules             *byte             // const char* tensor_type_rules;
	VAEDecodeOnly               uint8             // bool vae_decode_only;
	FreeParamsImmediately       uint8             // bool free_params_immediately;
	NThreads                    int32             // int n_threads;
	WType                       SDType            // enum sd_type_t wtype;
	RNG                         RNGType           // enum rng_type_t rng_type;
	SamplerRNG                  RNGType           // enum rng_type_t sampler_rng_type;
	Prediction                  PredictionType    // enum prediction_t prediction;
	LoraApplyMode               LoraApplyModeType // enum lora_apply_mode_t lora_apply_mode;
	OffloadParamsToCPU          uint8             // bool offload_params_to_cpu;
	EnableMMAP                  uint8             // bool enable_mmap;
	KeepClipOnCPU               uint8             // bool keep_clip_on_cpu;
	KeepControlNetOnCPU         uint8             // bool keep_control_net_on_cpu;
	KeepVAEOnCPU                uint8             // bool keep_vae_on_cpu;
	FlashAttn                   uint8             // bool flash_attn;
	DiffusionFlashAttn          uint8             // bool diffusion_flash_attn;
	TAEPreviewOnly              uint8             // bool tae_preview_only;
	DiffusionConvDirect         uint8             // bool diffusion_conv_direct;
	VAEConvDirect               uint8             // bool vae_conv_direct;
	CircularX                   uint8             // bool circular_x;
	CircularY                   uint8             // bool circular_y;
	ForceSDXLVAEConvScale       uint8             // bool force_sdxl_vae_conv_scale;
	ChromaUseDITMask            uint8             // bool chroma_use_dit_mask;
	ChromaUseT5Mask             uint8             // bool chroma_use_t5_mask;
	ChromaT5MaskPad             int32             // int chroma_t5_mask_pad;
	QwenImageZeroCond           uint8             // bool qwen_image_zero_cond_t;
}

// save to PNG
func utilsSaveImage(imgData *Image, filename string) error {
	if filename == "" {
		filename = "output.png"
	}
	width := int(imgData.Width)
	height := int(imgData.Height)

	// Create a new Go image container
	// Note: If your data is RGB, you'll need to map 3 bytes to 4 (RGBA)
	// or use a custom loop.
	rect := image.Rect(0, 0, width, height)
	rgba := image.NewRGBA(rect)

	// Convert *uint8 to a Go slice for easier handling
	// This assumes the data is packed R, G, B, R, G, B...
	pix := unsafe.Slice(imgData.Data, width*height*3)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			i := (y*width + x) * 3
			rgba.Set(x, y, color.RGBA{
				R: pix[i],
				G: pix[i+1],
				B: pix[i+2],
				A: 255, // Set full opacity
			})
		}
	}

	// Create the file and encode
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, rgba)
}

// creates C,Cpp-compatible empty string
func utilsGetNulString() *byte {
	s := []byte("\x00")
	return &s[0]
}

func utilsStrToNulString(text string) *byte {
	s := []byte(text + "\x00")
	return &s[0]
}

func NewContextParams() *ContextParams {
	ctxParams := ContextParams{
		ModelPath:                   utilsGetNulString(),
		ClipLPath:                   utilsGetNulString(),
		ClipGPath:                   utilsGetNulString(),
		ClipVisionPath:              utilsGetNulString(),
		T5XXLPath:                   utilsGetNulString(),
		LLMPath:                     utilsGetNulString(),
		LLMVisionPath:               utilsGetNulString(),
		DiffusionModelPath:          utilsGetNulString(),
		HighNoiseDiffusionModelPath: utilsGetNulString(),
		VAEPath:                     utilsGetNulString(),
		TAESDPath:                   utilsGetNulString(),
		ControlNetPath:              utilsGetNulString(),
		PhotoMakerPath:              utilsGetNulString(),
		TensorTypeRules:             utilsGetNulString(),
	}
	return &ctxParams
}

func DefaultContextParams(cp *ContextParams) *ContextParams {
	cp.DiffusionModelPath = utilsStrToNulString("/home/dom-ak45/.cache/stable.diffusion/flux-2-klein-9b-Q8_0.gguf")
	cp.LLMPath = utilsStrToNulString("/home/dom-ak45/.cache/stable.diffusion/Qwen3-8B-Q8_0.gguf")
	cp.VAEPath = utilsStrToNulString("/home/dom-ak45/.cache/stable.diffusion/diffusion_pytorch_model.safetensors")
	cp.KeepVAEOnCPU = 1
	return cp
}

type LoraApplyModeType int32

const (
	LoraApplyAuto LoraApplyModeType = iota
	LoraApplyImmediately
	LoraApplyAtRuntime
	LoraApplyModeCount
)

type PredictionType int32

const (
	EPSPred PredictionType = iota
	VPred
	EDMVPred
	FLOWPred
	FLUXFLOWPred
	FLUX2FLOWPred
	PredictionCount
)

type RNGType int32

const (
	STDDefaultRNG RNGType = iota
	CUDARNG
	CPURNG
	RNGTypeCount
)

type SDType int32

// same as enum ggml_type
const (
	TypeF32  SDType = 0
	TypeF16         = 1
	TypeQ4_0        = 2
	TypeQ4_1        = 3
	// SD_TYPE_Q4_2 = 4, support has been removed
	// SD_TYPE_Q4_3 = 5, support has been removed
	TypeQ5_0    = 6
	TypeQ5_1    = 7
	TypeQ8_0    = 8
	TypeQ8_1    = 9
	TypeQ2_K    = 10
	TypeQ3_K    = 11
	TypeQ4_K    = 12
	TypeQ5_K    = 13
	TypeQ6_K    = 14
	TypeQ8_K    = 15
	TypeIQ2_XXS = 16
	TypeIQ2_XS  = 17
	TypeIQ3_XXS = 18
	TypeIQ1_S   = 19
	TypeIQ4_NL  = 20
	TypeIQ3_S   = 21
	TypeIQ2_S   = 22
	TypeIQ4_XS  = 23
	TypeI8      = 24
	TypeI16     = 25
	TypeI32     = 26
	TypeI64     = 27
	TypeF64     = 28
	TypeIQ1_M   = 29
	TypeBF16    = 30
	// SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
	// SD_TYPE_Q4_0_4_8 = 32,
	// SD_TYPE_Q4_0_8_8 = 33,
	TypeTQ1_0 = 34
	TypeTQ2_0 = 35
	// SD_TYPE_IQ4_NL_4_4 = 36,
	// SD_TYPE_IQ4_NL_4_8 = 37,
	// SD_TYPE_IQ4_NL_8_8 = 38,
	TypeMXFP4 = 39 // MXFP4 (1 block)
	TypeCOUNT = 40
)

type Embedding struct {
	Name *byte
	Path *byte
}

var (
	// SD_API const char* sd_get_system_info();
	getSystemInfo ffi.Fun

	// SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);
	generateImage ffi.Fun

	// SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
	newContext ffi.Fun

	// SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
	imageGenParamsInit ffi.Fun

	printMyStruct ffi.Fun

	// SD_API void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);
	contextParamsInit ffi.Fun

	// SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
	setLogCallback ffi.Fun
)

// sd_lora_t
type LoraType struct {
	IsHighNoise uint8   // bool is_high_noise;
	Multiplier  float32 // float multiplier;
	Path        *byte   // const char* path;
}

type Image struct {
	Width   uint32 // uint32_t width
	Height  uint32 // uint32_t height
	Channel uint32 // uint32_t channel
	Data    *uint8 // uint8_t  *data
}

type SchedulerType int32

const (
	DiscreteScheduler SchedulerType = iota
	KarrasScheduler
	ExponentialScheduler
	AYSScheduler
	GITSScheduler
	SGMUniformScheduler
	SimpleScheduler
	SmoothStepScheduler
	KLOptimalScheduler
	LCMScheduler
	BongTangentScheduler
	SchedulerCount
)

type SampleMethodType int32

const (
	EulerSampleMethod SampleMethodType = iota
	EulerASampleMethod
	HeunSampleMethod
	DPM2SampleMethod
	DPMPP2SASampleMethod
	DPMPP2MSampleMethod
	DPMPP2Mv2SampleMethod
	IPNDMSampleMethod
	IPNDMVSampleMethod
	LCMSampleMethod
	DDIMTrailingSampleMethod
	TCDSampleMethod
	RESMultistepSampleMethod
	RES2SSampleMethod
	SampleMethodCount
)

type GuidanceParams struct {
	TextCfg           float32   // float txt_cfg;
	ImageCfg          float32   // float img_cfg;
	DistilledGuidance float32   // float distilled_guidance;
	SLG               SLGParams // sd_slg_params_t slg;
}

type SLGParams struct {
	Layers     *int32  // int* layers;
	LayerCount uint64  // size_t layer_count;
	LayerStart float32 // float layer_start;
	LayerEnd   float32 // float layer_end;
	Scale      float32 // float scale;
}

type SampleParamsType struct {
	Guidance          GuidanceParams   // sd_guidance_params_t guidance;
	Scheduler         SchedulerType    // enum scheduler_t scheduler;
	SampleMethod      SampleMethodType // enum sample_method_t sample_method;
	SampleSteps       int32            // int sample_steps;
	ETA               float32          // float eta;
	ShiftedTimestamp  int32            // int shifted_timestep;
	CustomSigmas      *float32         // float* custom_sigmas;
	CustomSigmasCount int32            // int custom_sigmas_count;
	FlowShift         float32          // float flow_shift;
}

type PMParamsType struct {
	IDImages      *Image  // sd_image_t* id_images;
	IDImagesCount int32   // int id_images_count;
	IDEmbedPath   *byte   // const char* id_embed_path;
	StyleStrength float32 // float style_strength;
}

type VAETilingParams struct {
	Enabled       uint8   // bool enabled;
	TileSizeX     int32   // int tile_size_x;
	TileSizeY     int32   // int tile_size_y;
	TargetOverlap float32 // float target_overlap;
	RelSizeX      float32 // float rel_size_x;
	RelSizeY      float32 // float rel_size_y;
}

type CacheModeType int32

const (
	CacheDisabled CacheModeType = iota
	CacheEasyCache
	CacheUcache
	CacheDBcache
	CacheTaylorseer
	CacheCacheDit
)

type CacheParams struct {
	Mode                     CacheModeType // enum sd_cache_mode_t mode;
	ReuseThreshold           float32       // float reuse_threshold;
	StartPercent             float32       // float start_percent;
	EndPercent               float32       // float end_percent;
	ErrorDecayRate           float32       // float error_decay_rate;
	UseRelativeThreshold     uint8         // bool use_relative_threshold;
	ResetErrorOnCompute      uint8         // bool reset_error_on_compute;
	FNComputeBlocks          int32         // int Fn_compute_blocks;
	BNComputeBlocks          int32         // int Bn_compute_blocks;
	ResidualDiffThreshold    float32       // float residual_diff_threshold;
	MaxWarmupSteps           int32         // int max_warmup_steps;
	MaxCachedSteps           int32         // int max_cached_steps;
	MaxContinuousCachedSteps int32         // int max_continuous_cached_steps;
	TaylorSeerNDerivatives   int32         // int taylorseer_n_derivatives;
	TaylorSeerSkipInterval   int32         // int taylorseer_skip_interval;
	SCMMask                  *byte         // const char* scm_mask;
	SCMPolicyDynamic         uint8         // bool scm_policy_dynamic;
}

type ImageParams struct {
	Lora               *LoraType        // const sd_lora_t* loras;
	LoraCount          uint32           // uint32_t lora_count;
	Prompt             *byte            // const char* prompt;
	NegativePrompt     *byte            // const char* negative_prompt;
	ClipSkip           int32            // int clip_skip;
	InitImage          Image            // sd_image_t init_image;
	RefImages          *Image           // sd_image_t* ref_images;
	RefImagesCount     int32            // int ref_images_count;
	AutoResizeRefImage uint8            // bool auto_resize_ref_image;
	IncreaseRefIndex   uint8            // bool increase_ref_index;
	MaskImage          Image            // sd_image_t mask_image;
	Width              int32            // int width;
	Height             int32            // int height;
	SampleParams       SampleParamsType // sd_sample_params_t sample_params;
	Strength           float32          // float strength;
	Seed               int64            // int64_t seed;
	BatchCount         int32            // int batch_count;
	ControlImage       Image            // sd_image_t control_image;
	ControlStrength    float32          // float control_strength;
	PMParams           PMParamsType     // sd_pm_params_t pm_params;
	VAETilingParams    VAETilingParams  // sd_tiling_params_t vae_tiling_params;
	Cache              CacheParams      // sd_cache_params_t cache;
}

func NewImageParams() *ImageParams {
	// TODO: resolve this with reflect
	ip := &ImageParams{
		Prompt:         utilsGetNulString(),
		NegativePrompt: utilsGetNulString(),
	}

	return ip
}

type MyStruct struct {
	LoraCount uint32
}

func DefaultImageParams(ip *ImageParams) *ImageParams {
	negPrompt := "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
	ip.NegativePrompt = utilsStrToNulString(negPrompt)

	prompt := "Ocean, sunset and palms beach."
	prompt = "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic."
	prompt = "Sunshine, nature, wind and smell of fall. The kids are playing around fire, parents chop woood and work. Melancholic feeling."

	prompt = "Abstract architectural visualization of a high-performance software library written in Go and C++. A dark, expansive grid platform where glowing cubes representing individual AI model containers are being 'plugged in.' Inside each cube, vibrant holographic icons indicate the model type (text, image, audio). Data lines pulse between the cubes like fiber optics. Cubes represent models, eg. LLM, Stable diffusion, audio etc. both textually and visually. Cyberpunk aesthetic, sharp lines, hyper-realistic textures, isometric view, soft ambient neon glow."

	prompt = "Sand beach with green sand, an ocean with red color, lightning strikings and wind blows. Ultra realistic, no CGI."

	prompt = "A hyper-realistic, wide-angle shot of a Victorian-era clockwork robot wearing a 'Go' t-shirt, frantically trying to plug glowing blue fiber-optic cables into a giant, steam-powered wooden server rack in a library filled with floating books. 8k resolution, cinematic lighting, highly detailed brass gears."

	prompt = "A group of highly detailed squirrels in tiny business suits playing a high-stakes game of poker around a hollowed-out tree stump. One squirrel is wearing a monocle and looking suspiciously at a 'Go' gopher mascot sitting at the table. Cinematic smoke, moody lighting, oil painting style. 8k resolution, cinematic lighting."

	prompt = "A T-Rex in a tiny, neon-lit 1950s diner trying to eat a comically small cupcake with giant silver tweezers. The T-Rex is wearing a pink apron. Cyberpunk aesthetic, vibrant neon colors, reflection on wet floor, 4k digital art."

	// prompt = "An octopus DJing at an underwater rave inside a sunken pirate ship. Eight tentacles operating glowing DJ decks made of bioluminescent jellyfish. Sea turtles wearing neon glow-sticks in the background. Hyper-realistic, vibrant blue and purple lighting, bubbles, sharp focus."
	// prompt = "a medieval knight debugging code on a glowing laptop, rubber ducks covering the desk, dramatic torch lighting, castle server room, cinematic, highly detailed"

	// prompt = "a very serious orange cat wearing glasses presenting a complicated architecture diagram to confused mice in a conference room, whiteboard full of spaghetti code arrows, professional presentation lighting"

	// prompt = "an ancient wizard maintaining an open source project, magical terminal floating in the air, contributors arriving as glowing pull requests, fantasy library background, epic lighting"

	// prompt = "robots sitting in a classroom while a tired human teacher explains what a captcha is, chalkboard that says \"prove you are not a robot\", comedic scene, detailed illustration"

	// prompt = "a raccoon CEO pitching a startup idea to a room full of skeptical venture capitalist pigeons, slide on screen says \"AI-powered trash optimization\", corporate meeting room"

	// prompt = "two programmers in wizard robes fighting a magical duel made of git merge conflicts, glowing text like <<<<<<< HEAD floating in the air, fantasy battle scene"

	ip.Prompt = utilsStrToNulString(prompt)

	ip.SampleParams.Guidance.TextCfg = 5.0

	ip.SampleParams.SampleSteps = 10

	ip.Width = 900
	ip.Height = 900

	return ip
}

// try at bindings implementation of stable-diffusion.cpp lib

func loadError(name string, err error) error {
	fmt.Println(fmt.Errorf("could not load %q: %w", name, err).Error())
	return fmt.Errorf("could not load %q: %w", name, err)
}

func LoadLibrary(path, lib string) (ffi.Lib, error) {
	if path == "" && os.Getenv("SD_DYN_LIB") != "" {
		path = os.Getenv("SD_DYN_LIB")
	}
	if path == "" {
		return ffi.Lib{}, fmt.Errorf("Can't find runtime stable-diffusion libraries")
	}

	filename := GetLibraryFilename(path, lib)
	return ffi.Load(filename)
}

// fetches the .so lib created (and named!) by the stable-diffusion cmake build
func GetLibraryFilename(path, lib string) string {
	switch runtime.GOOS {
	case "linux", "freebsd":
		return filepath.Join(path, fmt.Sprintf("lib%s.so", lib))
	default:
		panic(fmt.Sprintf("OS %q not supported", runtime.GOOS))
	}
}

func loadDS(lib ffi.Lib) error {
	var err error
	if getSystemInfo, err = lib.Prep("sd_get_system_info", &ffi.TypePointer); err != nil {
		return loadError("sd_get_system_info", err)
	}
	if generateImage, err = lib.Prep("generate_image", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("generate_image", err)
	}
	if newContext, err = lib.Prep("new_sd_ctx", &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("new_sd_ctx", err)
	}
	if imageGenParamsInit, err = lib.Prep("sd_img_gen_params_init", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("sd_img_gen_params_init", err)
	}
	if printMyStruct, err = lib.Prep("PrintMyStruct", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("PrintMyStruct", err)
	}
	if contextParamsInit, err = lib.Prep("sd_ctx_params_init", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("sd_ctx_params_init", err)
	}

	if setLogCallback, err = lib.Prep("sd_set_log_callback", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("sd_set_log_callback", err)
	}

	fmt.Println("SUCCESS")
	return nil
}

func GetSystemInfo() string {
	var systemInfo *byte
	getSystemInfo.Call(&systemInfo)

	if systemInfo == nil {
		return ""
	}

	fmt.Println(unix.BytePtrToString(systemInfo))
	return string(*systemInfo)
}

func GenerateImage(ctx Context, ip ImageParams) {
	var image *Image

	i := &ip
	generateImage.Call(unsafe.Pointer(&image), unsafe.Pointer(&ctx), unsafe.Pointer(&i))
	fmt.Printf("%#v", *image)
	utilsSaveImage(image, "")
	// fmt.Printf("%#v", image.Width)
	// fmt.Printf("%#v", image.Height)
	// fmt.Printf("%#v", image.Channel)
	// fmt.Printf("%#v", image.Data)
}

func NewContext(ctxParams ContextParams) Context {
	var context Context

	ctx := &ctxParams
	fmt.Println(*ctx)
	newContext.Call(unsafe.Pointer(&context), unsafe.Pointer(&ctx))

	fmt.Println(context)
	return context
}

func PrintMyStruct() {
	var ms MyStruct
	ms.LoraCount = 10
	printMyStruct.Call(nil, unsafe.Pointer(&ms))
}

func ImageGenParamsInit() ImageParams {
	var ip *ImageParams = NewImageParams()

	imageGenParamsInit.Call(nil, unsafe.Pointer(&ip))
	// c, _ := json.Marshal(*ip)
	// fmt.Println(string(c))
	// fmt.Printf("%#v\n", ip.SampleParams)
	// fmt.Printf("%#v\n", ip.InitImage)
	// fmt.Printf("%#v\n", ip.PMParams)
	// fmt.Printf("%#v\n", ip.VAETilingParams)
	// fmt.Printf("%#v\n", ip)
	return *ip
}

// Creates default context params
func ContextParamsInit() ContextParams {
	var cp *ContextParams = NewContextParams()

	contextParamsInit.Call(nil, unsafe.Pointer(&cp))
	// fmt.Println("HERE DOWN")
	// fmt.Printf("%#v\n", cp.VAEDecodeOnly)
	// fmt.Printf("%#v\n", cp.FreeParamsImmediately)
	// fmt.Printf("%#v\n", cp.NThreads)
	// fmt.Printf("%#v\n", cp.WType)
	// fmt.Printf("%#v\n", cp.RNG)
	// fmt.Printf("%#v\n", cp.SamplerRNG)
	// fmt.Printf("%#v\n", cp.Prediction)
	// fmt.Printf("%#v\n", cp.LoraApplyMode)
	// fmt.Printf("%#v\n", cp.OffloadParamsToCPU)
	// fmt.Printf("%#v\n", cp.EnableMMAP)
	// fmt.Printf("%#v\n", cp.KeepClipOnCPU)
	// fmt.Printf("%#v\n", cp.KeepControlNetOnCPU)

	// fmt.Printf("%#v\n", cp.KeepVAEOnCPU)
	// fmt.Printf("%#v\n", cp.DiffusionFlashAttn)
	// fmt.Printf("%#v\n", cp.CircularX)
	// fmt.Printf("%#v\n", cp.CircularY)
	// fmt.Printf("%#v\n", cp.ChromaUseDITMask)
	// fmt.Printf("%#v\n", cp.ChromaUseT5Mask)
	// fmt.Printf("%#v\n", cp.ChromaT5MaskPad)
	return *cp
}

type LogLevel int32

const (
	Debug LogLevel = iota
	Info
	Warn
	Error
)

type LogCallback func(level LogLevel, text string, data unsafe.Pointer)

var progressCallback unsafe.Pointer
var sizeOfClosure = unsafe.Sizeof(ffi.Closure{})

func SetLogCallback(callback LogCallback, data unsafe.Pointer) {
	if callback == nil {
		panic("Can't set nil as a callback")
	}

	closure := ffi.ClosureAlloc(sizeOfClosure, &progressCallback)

	fn := ffi.NewCallback(func(cif *ffi.Cif, ret unsafe.Pointer, args *unsafe.Pointer, userData unsafe.Pointer) uintptr {
		if args == nil {
			return 1 // error
		}

		arg := unsafe.Slice(args, cif.NArgs)

		level := *(*LogLevel)(arg[0])
		text := *(**byte)(arg[1])
		data := *(*unsafe.Pointer)(arg[2])

		callback(
			level,
			unix.BytePtrToString(text),
			data,
		)
		return 0
	})

	var cifCallback ffi.Cif
	if status := ffi.PrepCif(&cifCallback, ffi.DefaultAbi, 3, &ffi.TypeVoid, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer); status != ffi.OK {
		panic(status)
	}

	if closure != nil {
		if status := ffi.PrepClosureLoc(closure, &cifCallback, fn, nil, progressCallback); status != ffi.OK {
			panic(status)
		}
	}

	setLogCallback.Call(nil, &progressCallback, unsafe.Pointer(&data))

}

// Load loads the shared llama.cpp libraries from the specified path.
func Load(path string) error {
	lib, err := LoadLibrary(path, "stable-diffusion")
	if err != nil {
		fmt.Println(loadError("stable-diffusion", err).Error())
		return err
	}
	loadDS(lib)

	// lib, err = loader.LoadLibrary(path, "ggml-base")
	// if err != nil {
	// 	return err
	// }

	// if err := loadGGMLBase(lib); err != nil {
	// 	return err
	// }

	// lib, err = loader.LoadLibrary(path, "llama")
	// if err != nil {
	// 	return err

	// }
	return nil
}

var logCallback LogCallback = func(level LogLevel, text string, data unsafe.Pointer) {
	fmt.Println("GO CALLBACK: ")
	fmt.Println("level: ", level)
	fmt.Println("text: ", text)
	data2 := *(*struct{ text string })(data)
	fmt.Println("data: ", data2)
}

func main() {
	Load("")
	var ctx Context

	GetSystemInfo()

	ctxtParams := ContextParamsInit()
	DefaultContextParams(&ctxtParams)
	ctx = NewContext(ctxtParams)

	imgParams := ImageGenParamsInit()
	DefaultImageParams(&imgParams)
	var data = &struct{ text string }{text: "dodo kralj"}

	SetLogCallback(logCallback, unsafe.Pointer(data))

	GenerateImage(ctx, imgParams)
}
